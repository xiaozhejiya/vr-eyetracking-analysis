import os
import re
import sys
import numpy as np
import pandas as pd

GROUP_TYPES_DEFAULT = ["control", "mci", "ad"]

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def import_event_analyzer():
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()

def calibration_output_dir():
    return os.path.join(project_root(), "lsh_eye_analysis", "data_calibartion")

def calibration_output_path(file_path):
    base = os.path.basename(file_path).replace("_preprocessed.csv", "_preprocessed_calibrated.csv")
    data_root = data_dir()
    try:
        rel = os.path.relpath(file_path, data_root)
    except ValueError:
        rel = None
    group = None
    if rel:
        parts = rel.split(os.sep)
        if len(parts) >= 1 and parts[0].endswith("_processed"):
            group = parts[0].replace("_processed", "")
    if group:
        out_dir = os.path.join(calibration_output_dir(), group)
    else:
        out_dir = calibration_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)

def data_dir(*parts):
    """拼接 `data/` 子目录的路径，例如 `data_dir('control_processed')`"""
    return os.path.join(project_root(), "data", *parts)


def group_root(group):
    return data_dir(f"{group}_processed")


def list_subject_folders(group):
    root = group_root(group)
    if not os.path.exists(root):
        return []
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def calibrate_groups(groups, dx_bounds=(-0.05, 0.05), dy_bounds=(-0.05, 0.05), step=0.005, weights=None, apply=False):
    results = []
    for g in groups:
        folders = list_subject_folders(g)
        for folder in folders:
            res = calibrate_subject_folder(folder, dx_bounds=dx_bounds, dy_bounds=dy_bounds, step=step, weights=weights, apply=apply)
            results.append({"group": g, "folder": folder, "results": res})
    return results

def parse_q_num_from_filename(filename):
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1

def _get_dt(df):
    if "time_diff" in df.columns:
        dt = df["time_diff"].to_numpy()
        return np.where(np.isnan(dt), 0.0, dt)
    if "milliseconds" in df.columns:
        ms = df["milliseconds"].to_numpy()
        d = np.diff(ms, prepend=ms[0])
        return np.where(d < 0, 0.0, d)
    return np.ones(len(df), dtype=float)

def _in_any_roi(xs, ys, roi_list):
    if not roi_list:
        return np.zeros_like(xs, dtype=bool)
    inside = np.zeros_like(xs, dtype=bool)
    for _, xmn, ymn, xmx, ymy in roi_list:
        inside |= (xs >= xmn) & (xs <= xmx) & (ys >= ymn) & (ys <= ymy)
    return inside

def _enter_count(mask):
    if len(mask) < 2:
        return 0
    prev = mask[:-1]
    curr = mask[1:]
    return int(np.sum((~prev) & curr))

def apply_offset(df, dx, dy):
    out = df.copy()
    if "x" not in out.columns or "y" not in out.columns:
        return out
    xs = out["x"].to_numpy(dtype=float)
    ys = out["y"].to_numpy(dtype=float)
    xs = np.clip(xs + dx, 0.0, 1.0)
    ys = np.clip(ys + dy, 0.0, 1.0)
    out["x"] = xs
    out["y"] = ys
    return out

def optimize_offset_by_roi(df, roi_kw, roi_inst, roi_bg, dx_bounds=(-0.05, 0.05), dy_bounds=(-0.05, 0.05), step=0.005, weights=None):
    if weights is None:
        weights = {
            "inst_time": 1.0,
            "kw_time": 0.3,
            "bg_time": 0.5,
            "inst_enter": 0.0,
            "kw_enter": 0.0,
            "bg_enter": 0.0,
        }
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}
    xs0 = df["x"].to_numpy(dtype=float)
    ys0 = df["y"].to_numpy(dtype=float)
    dt = _get_dt(df)

    best = {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}
    dx_vals = np.arange(dx_bounds[0], dx_bounds[1] + 1e-12, step)
    dy_vals = np.arange(dy_bounds[0], dy_bounds[1] + 1e-12, step)

    for dx in dx_vals:
        for dy in dy_vals:
            xs = np.clip(xs0 + dx, 0.0, 1.0)
            ys = np.clip(ys0 + dy, 0.0, 1.0)

            m_kw = _in_any_roi(xs, ys, roi_kw)
            m_inst = _in_any_roi(xs, ys, roi_inst)
            m_bg = _in_any_roi(xs, ys, roi_bg)

            t_kw = float(np.sum(dt[m_kw])) if len(dt) else 0.0
            t_inst = float(np.sum(dt[m_inst])) if len(dt) else 0.0
            t_bg = float(np.sum(dt[m_bg])) if len(dt) else 0.0

            e_kw = _enter_count(m_kw)
            e_inst = _enter_count(m_inst)
            e_bg = _enter_count(m_bg)

            score = (
                weights["inst_time"] * t_inst
                + weights["kw_time"] * t_kw
                - weights["bg_time"] * t_bg
                + weights["inst_enter"] * e_inst
                + weights["kw_enter"] * e_kw
                - weights["bg_enter"] * e_bg
            )

            if score > best["score"]:
                best["dx"] = float(dx)
                best["dy"] = float(dy)
                best["score"] = float(score)
                best["metrics"] = {
                    "inst_time": t_inst,
                    "kw_time": t_kw,
                    "bg_time": t_bg,
                    "inst_enter": int(e_inst),
                    "kw_enter": int(e_kw),
                    "bg_enter": int(e_bg),
                }

    return best

def calibrate_file_by_roi_grid(file_path, dx_bounds=(-0.05, 0.05), dy_bounds=(-0.05, 0.05), step=0.005, weights=None, apply=False, save_path=None):
    df = pd.read_csv(file_path)
    q = parse_q_num_from_filename(os.path.basename(file_path))
    analyzer = import_event_analyzer()
    kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
    best = optimize_offset_by_roi(df, kw, inst, bg, dx_bounds=dx_bounds, dy_bounds=dy_bounds, step=step, weights=weights)
    applied_path = None
    if apply:
        out_df = apply_offset(df, best["dx"], best["dy"])
        if save_path is None:
            save_path = calibration_output_path(file_path)
        out_df.to_csv(save_path, index=False)
        applied_path = save_path
    return {"file_path": file_path, "q": q, "best": best, "applied_path": applied_path}

def calibrate_subject_folder(folder, dx_bounds=(-0.05, 0.05), dy_bounds=(-0.05, 0.05), step=0.005, weights=None, apply=False):
    results = []
    for name in os.listdir(folder):
        if name.endswith("_preprocessed.csv"):
            fp = os.path.join(folder, name)
            res = calibrate_file_by_roi_grid(fp, dx_bounds=dx_bounds, dy_bounds=dy_bounds, step=step, weights=weights, apply=apply)
            results.append(res)
    return results

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="ROI-driven grid calibration")
    parser.add_argument("--file", type=str, help="path to a single _preprocessed.csv file")
    parser.add_argument("--folder", type=str, help="path to a subject folder containing _preprocessed.csv files")
    parser.add_argument("--groups", type=str, default=None, help="comma-separated group types, e.g. control,sci,ad")
    parser.add_argument("--dx-min", type=float, default=-0.05)
    parser.add_argument("--dx-max", type=float, default=0.05)
    parser.add_argument("--dy-min", type=float, default=-0.05)
    parser.add_argument("--dy-max", type=float, default=0.05)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--weights", type=str, default=None, help="JSON string of weights, e.g. {\"inst_time\":1.2,\"bg_time\":0.8}")
    parser.add_argument("--no-apply", action="store_true", help="do not write calibrated CSVs")
    args = parser.parse_args()

    apply_flag = not args.no_apply
    weights_obj = None
    if args.weights:
        try:
            weights_obj = json.loads(args.weights)
        except Exception:
            weights_obj = None

    if args.folder:
        result = calibrate_subject_folder(
            args.folder,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.file:
        result = calibrate_file_by_roi_grid(
            args.file,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.groups:
        groups = [s.strip() for s in args.groups.split(",") if s.strip()]
        result = calibrate_groups(
            groups,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = calibrate_groups(
            GROUP_TYPES_DEFAULT,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))