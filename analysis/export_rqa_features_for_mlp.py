import os
import re
import sys
import pandas as pd
import numpy as np
from typing import List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import Config

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def parse_q_num_from_filename(filename: str) -> int:
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1


def list_subject_folders_calibrated(root: str) -> List[str]:
    if not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def list_q_files(folder: str) -> List[str]:
    files = []
    if not os.path.exists(folder):
        return files
    for name in os.listdir(folder):
        if name.endswith("_preprocessed_calibrated.csv"):
            files.append(os.path.join(folder, name))
    return files


def group_calibrated_root(calib_dir_name: str, group: str) -> str:
    return os.path.join(project_root(), "data", calib_dir_name, f"{group}_calibrated")


def export_group_rqa_features(group: str, calib_dir_name: str, params: Dict) -> Dict[int, str]:
    sys.path.append(project_root())
    from analysis.rqa_analyzer import RQAAnalyzer

    rqa = RQAAnalyzer()
    q_rows: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    root = group_calibrated_root(calib_dir_name, group)
    subjects = list_subject_folders_calibrated(root)

    for subj in subjects:
        subj_name = os.path.basename(subj)
        for fp in list_q_files(subj):
            q = parse_q_num_from_filename(os.path.basename(fp))
            if q not in q_rows:
                continue
            res = rqa.analyze_single_file(fp, params)
            row = {
                "group": group,
                "subject": subj_name,
                "q": q,
                "RR_1D_x": float(res.get("RR-1D-x", np.nan)),
                "DET_1D_x": float(res.get("DET-1D-x", np.nan)),
                "ENT_1D_x": float(res.get("ENT-1D-x", np.nan)),
                "RR_2D_xy": float(res.get("RR-2D-xy", np.nan)),
                "DET_2D_xy": float(res.get("DET-2D-xy", np.nan)),
                "ENT_2D_xy": float(res.get("ENT-2D-xy", np.nan)),
            }
            q_rows[q].append(row)

    out_dir = ensure_dir(os.path.join(project_root(), "data", "MLP_data", "features", "rqa_features", f"{group}_group"))
    saved: Dict[int, str] = {}
    for q in [1, 2, 3, 4, 5]:
        rows = q_rows[q]
        if rows:
            df = pd.DataFrame(rows)
            # 若重复（同一subject同一题号有多个文件），按subject聚合为均值
            df = (
                df.groupby(["group", "subject", "q"], as_index=False)
                  .agg({
                      "RR_1D_x": "mean",
                      "DET_1D_x": "mean",
                      "ENT_1D_x": "mean",
                      "RR_2D_xy": "mean",
                      "DET_2D_xy": "mean",
                      "ENT_2D_xy": "mean",
                  })
            )
        else:
            df = pd.DataFrame(columns=[
                "group", "subject", "q",
                "RR_1D_x", "DET_1D_x", "ENT_1D_x",
                "RR_2D_xy", "DET_2D_xy", "ENT_2D_xy",
            ])
        out_fp = os.path.join(out_dir, f"{group}_q{q}.csv")
        df.to_csv(out_fp, index=False)
        saved[q] = out_fp
    return saved


def export_all(groups: List[str], calib_dir_name: str, params: Dict) -> Dict[str, Dict[int, str]]:
    res: Dict[str, Dict[int, str]] = {}
    for g in groups:
        res[g] = export_group_rqa_features(g, calib_dir_name, params)
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export RQA features per group/question to CSV for MLP")
    parser.add_argument("--groups", type=str, default="control,mci,ad")
    parser.add_argument("--calibrated-dir-name", type=str, default="data_calibration_mix",
                        help="data_calibration | data_calibration_mix | data_calibration_gradient_ascent")
    parser.add_argument("--m", type=int, default=Config.RQA_DEFAULT_PARAMS['m'])
    parser.add_argument("--tau", type=int, default=Config.RQA_DEFAULT_PARAMS['tau'])
    parser.add_argument("--eps", type=float, default=Config.RQA_DEFAULT_PARAMS['eps'])
    parser.add_argument("--lmin", type=int, default=Config.RQA_DEFAULT_PARAMS['lmin'])
    args = parser.parse_args()

    groups = [s.strip() for s in args.groups.split(",") if s.strip()]
    def clamp(name, value):
        r = Config.RQA_PARAM_RANGES[name]
        return max(r['min'], min(r['max'], value))
    params = {
        "m": clamp('m', args.m),
        "tau": clamp('tau', args.tau),
        "eps": clamp('eps', args.eps),
        "lmin": clamp('lmin', args.lmin),
    }
    result = export_all(groups, args.calibrated_dir_name, params)
    print(result)

