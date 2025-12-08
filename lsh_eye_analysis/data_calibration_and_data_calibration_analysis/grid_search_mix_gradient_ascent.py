import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import json
import time

# 默认包含的被试分组名称（和 data/ 下面的目录对应）
GROUP_TYPES_DEFAULT = ["control", "mci", "ad"]


def project_root():
    """
    返回项目根目录：
    当前文件 -> 上级目录 -> 上上级目录
    data/, analysis/ 等目录都假定挂在这个根目录下。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add project root to sys.path to allow imports from lsh_eye_analysis
sys.path.append(project_root())
from lsh_eye_analysis.utils.score_function import calculate_score_and_metrics, apply_offset, calculate_score_grid, get_dt
from lsh_eye_analysis.utils.CosineWarmupDecay import CosineWarmupDecay


def import_event_analyzer():
    """
    动态导入 EventAnalyzer。
    """
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()


def calibration_output_dir():
    """
    所有校准后 CSV 的统一输出根目录。
    """
    return os.path.join(project_root(), "lsh_eye_analysis", "data", "data_calibration_mix")


def calibration_output_path(file_path):
    """
    根据原始文件路径构建校准结果的输出路径
    """
    base = os.path.basename(file_path).replace(
        "_preprocessed.csv", "_preprocessed_calibrated.csv"
    )

    data_root = data_dir()
    try:
        rel = os.path.relpath(file_path, data_root)
    except ValueError:
        rel = None

    group = None
    subject = None
    if rel:
        parts = rel.split(os.sep)
        if len(parts) >= 2 and parts[0].endswith("_processed"):
            group = parts[0].replace("_processed", "")
            subject = parts[1]

    if group:
        out_dir = os.path.join(calibration_output_dir(), f"{group}_calibrated")
        if subject:
            out_dir = os.path.join(out_dir, subject)
    else:
        out_dir = calibration_output_dir()

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


def data_dir(*parts):
    return os.path.join(project_root(), "data", *parts)


def group_root(group):
    return data_dir(f"{group}_processed")


def list_subject_folders(group):
    root = group_root(group)
    if not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def parse_q_num_from_filename(filename):
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1


def optimize_offset_mix(
    df,
    roi_kw,
    roi_inst,
    roi_bg,
    dx_bounds=(-0.1, 0.1),
    dy_bounds=(-0.1, 0.1),
    grid_step=0.01,
    weights=None,
    score_type="soft",
):
    """
    混合优化策略：
    1. 先使用 Grid Search 在粗网格上寻找全局最优初始点。
    2. 以该点为起点，使用 Gradient Ascent 进行精细优化。
    """
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}

    if weights is None:
        weights = {}

    print(f"  > Step 1: Grid Search (bounds={dx_bounds}, step={grid_step})...")
    # Step 1: Grid Search
    # 注意：Grid Search 可以使用 hard 或 soft，这里为了速度建议先用 hard 或低精度的 soft
    # 但为了梯度上升的连贯性，如果最终目标是 soft score，最好 Grid 也用 soft
    grid_res = calculate_score_grid(
        df,
        roi_kw,
        roi_inst,
        dx_bounds=dx_bounds,
        dy_bounds=dy_bounds,
        step=grid_step,
        weights=weights,
        score_type=score_type 
    )
    
    best_dx_grid = grid_res["dx"]
    best_dy_grid = grid_res["dy"]
    best_score_grid = grid_res["score"]
    print(f"    Grid result: dx={best_dx_grid:.4f}, dy={best_dy_grid:.4f}, score={best_score_grid:.4f}")

    # Step 2: Gradient Ascent
    # 使用 Grid Search 的结果作为初始值
    print(f"  > Step 2: Gradient Ascent fine-tuning...")
    
    # -----------------------------
    # 梯度上升逻辑 (复用自 data_calibration_gradient_ascent)
    # -----------------------------
    w_inst = float(weights.get("inst_time", 1.0))
    w_kw   = float(weights.get("kw_time", 1.0))
    w_bg   = float(weights.get("bg_time", 0.7))
    lambda_reg = float(weights.get("lambda_reg", 0.0))

    xs_np = df["x"].to_numpy(dtype=float)
    ys_np = df["y"].to_numpy(dtype=float)
    dt_np = get_dt(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(xs_np, dtype=torch.float32, device=device)
    y = torch.tensor(ys_np, dtype=torch.float32, device=device)
    dt = torch.tensor(dt_np, dtype=torch.float32, device=device)

    # 初始化为 Grid Search 的结果
    dx = torch.nn.Parameter(torch.tensor(best_dx_grid, dtype=torch.float32, device=device))
    dy = torch.nn.Parameter(torch.tensor(best_dy_grid, dtype=torch.float32, device=device))

    # 使用 CosineWarmupDecay 调度器
    lr = 0.01  # 初始学习率
    min_lr = 0.0005
    n_steps = 100
    warmup_step = 10
    
    optimizer = torch.optim.Adam([dx, dy], lr=lr)
    
    scheduler = CosineWarmupDecay(
        optimizer, 
        initial_lr=lr, 
        min_lr=min_lr, 
        warmup_step=warmup_step, 
        total_step=n_steps, 
        multi=0, 
        print_step=-1
    )

    # Soft ROI 函数 (Torch 版本)
    def soft_prob(xs, ys, roi_list, k=60.0):
        if not roi_list:
            return torch.zeros_like(xs)
        params = torch.tensor(
            [[xmn, ymn, xmx, ymy] for (_, xmn, ymn, xmx, ymy) in roi_list],
            dtype=xs.dtype,
            device=xs.device,
        )
        xmn = params[:, 0]
        ymn = params[:, 1]
        xmx = params[:, 2]
        ymy = params[:, 3]
        xs2 = xs.unsqueeze(1)
        ys2 = ys.unsqueeze(1)
        px1 = torch.sigmoid(k * (xs2 - xmn))
        px2 = torch.sigmoid(k * (xmx - xs2))
        py1 = torch.sigmoid(k * (ys2 - ymn))
        py2 = torch.sigmoid(k * (ymy - ys2))
        p = px1 * px2 * py1 * py2
        return torch.clamp(p.sum(dim=1), 0.0, 1.0)

    best_score_ga = best_score_grid # 初始 best score
    best_dx_ga = best_dx_grid
    best_dy_ga = best_dy_grid

    no_improve_count = 0
    patience = 20

    for step_idx in range(n_steps):
        optimizer.zero_grad()


        xs = torch.clamp(x + dx, 0.0, 1.0)
        ys = torch.clamp(y + dy, 0.0, 1.0)

        p_inst = soft_prob(xs, ys, roi_inst)
        p_kw   = soft_prob(xs, ys, roi_kw)

        p_pos = torch.clamp(p_inst + p_kw, 0.0, 1.0)
        p_bg  = 1.0 - p_pos

        T_inst = (dt * p_inst).sum()
        T_kw   = (dt * p_kw).sum()
        T_bg   = (dt * p_bg).sum()

        num   = w_inst * T_inst + w_kw * T_kw
        denom = num + w_bg * T_bg + 1e-6

        time_ratio = num / denom
        reg = lambda_reg * (dx * dx + dy * dy)
        score = time_ratio - reg

        loss = -score
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        with torch.no_grad():
            s_val = score.item()
            if s_val > best_score_ga:
                best_score_ga = s_val
                best_dx_ga = dx.item()
                best_dy_ga = dy.item()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f"    Early stopping at step {step_idx}: no improvement for {patience} steps.")
                break
    
    print(f"    GA result: dx={best_dx_ga:.4f}, dy={best_dy_ga:.4f}, score={best_score_ga:.4f}")

    # Final Calculation
    score, metrics = calculate_score_and_metrics(
        df, float(best_dx_ga), float(best_dy_ga), roi_kw, roi_inst, weights, score_type=score_type
    )

    best = {
        "dx": float(best_dx_ga),
        "dy": float(best_dy_ga),
        "score": float(score),
        "metrics": metrics,
    }

    return best


def calibrate_file_mix(
    file_path,
    dx_bounds=(-0.1, 0.1),
    dy_bounds=(-0.1, 0.1),
    grid_step=0.01,
    weights=None,
    apply=False,
    save_path=None,
    score_type="soft",
):
    df = pd.read_csv(file_path)
    q = parse_q_num_from_filename(os.path.basename(file_path))

    analyzer = import_event_analyzer()
    kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")

    best = optimize_offset_mix(
        df,
        kw,
        inst,
        bg,
        dx_bounds=dx_bounds,
        dy_bounds=dy_bounds,
        grid_step=grid_step,
        weights=weights,
        score_type=score_type,
    )

    applied_path = None
    if apply:
        out_df = apply_offset(df, best["dx"], best["dy"])
        if save_path is None:
            save_path = calibration_output_path(file_path)
        out_df.to_csv(save_path, index=False)
        applied_path = save_path

    return {"file_path": file_path, "q": q, "best": best, "applied_path": applied_path}


def calibrate_subject_folder_mix(
    folder,
    dx_bounds=(-0.1, 0.1),
    dy_bounds=(-0.1, 0.1),
    grid_step=0.01,
    weights=None,
    apply=False,
    score_type="soft",
):
    results = []
    for name in os.listdir(folder):
        if name.endswith("_preprocessed.csv"):
            fp = os.path.join(folder, name)
            print(f"Processing {name}...")
            res = calibrate_file_mix(
                fp,
                dx_bounds=dx_bounds,
                dy_bounds=dy_bounds,
                grid_step=grid_step,
                weights=weights,
                apply=apply,
                score_type=score_type,
            )
            results.append(res)
    return results


def calibrate_groups_mix(
    groups,
    dx_bounds=(-0.1, 0.1),
    dy_bounds=(-0.1, 0.1),
    grid_step=0.01,
    weights=None,
    apply=False,
    score_type="soft",
):
    results = []
    for g in groups:
        folders = list_subject_folders(g)
        for folder in folders:
            print(f"=== Processing Group: {g}, Subject: {os.path.basename(folder)} ===")
            res = calibrate_subject_folder_mix(
                folder,
                dx_bounds=dx_bounds,
                dy_bounds=dy_bounds,
                grid_step=grid_step,
                weights=weights,
                apply=apply,
                score_type=score_type,
            )
            results.append({"group": g, "folder": folder, "results": res})
    return results


def summarize_groups_score_speed_mix(
    groups,
    dx_bounds=(-0.1, 0.1),
    dy_bounds=(-0.1, 0.1),
    grid_step=0.01,
    weights=None,
    output_csv=None,
    score_type="soft",
):
    rows = []
    for g in groups:
        folders = list_subject_folders(g)
        scores_q = {1: [], 2: [], 3: [], 4: [], 5: []}
        proc_ms_q = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for folder in folders:
            for name in os.listdir(folder):
                if name.endswith("_preprocessed.csv"):
                    q = parse_q_num_from_filename(name)
                    if q not in (1, 2, 3, 4, 5):
                        continue
                    fp = os.path.join(folder, name)
                    
                    t0 = time.perf_counter()
                    res = calibrate_file_mix(
                        fp,
                        dx_bounds=dx_bounds,
                        dy_bounds=dy_bounds,
                        grid_step=grid_step,
                        weights=weights,
                        apply=False,
                        score_type=score_type,
                    )
                    t1 = time.perf_counter()
                    
                    proc_ms_q[q].append(float((t1 - t0) * 1000.0))
                    if res and res.get("best"):
                        scores_q[q].append(float(res["best"]["score"]))
        
        row = {"group": g}
        for q in range(1, 6):
            row[f"avg_score_q{q}"] = float(np.mean(scores_q[q])) if scores_q[q] else float("nan")
            row[f"avg_proc_ms_q{q}"] = float(np.mean(proc_ms_q[q])) if proc_ms_q[q] else float("nan")
        
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Reorder columns: group, scores (q1-q5), speeds (q1-q5)
    cols_order = ["group"]
    cols_order.extend([f"avg_score_q{q}" for q in range(1, 6)])
    cols_order.extend([f"avg_proc_ms_q{q}" for q in range(1, 6)])
    df_out = df_out.reindex(columns=cols_order)

    if output_csv is None:
        output_csv = os.path.join(calibration_output_dir(), "score_proc_summary_mix.csv")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return output_csv, df_out


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mixed Calibration: Grid Search + Gradient Ascent")
    parser.add_argument("--file", type=str, help="path to a single _preprocessed.csv file")
    parser.add_argument("--folder", type=str, help="path to a subject folder")
    parser.add_argument("--groups", type=str, default=None, help="comma-separated groups")
    
    parser.add_argument("--dx-min", type=float, default=-0.45)
    parser.add_argument("--dx-max", type=float, default=0.45)
    parser.add_argument("--dy-min", type=float, default=-0.45)
    parser.add_argument("--dy-max", type=float, default=0.45)
    parser.add_argument("--grid-step", type=float, default=0.05, help="Step size for initial grid search")
    
    parser.add_argument("--weights", type=str, default=None, help="JSON weights")
    parser.add_argument("--no-apply", action="store_true", help="Do not write output CSVs")
    parser.add_argument("--score-type", type=str, choices=["hard", "soft"], default="soft", help="Score type")
    
    parser.add_argument("--summary-score-speed", action="store_true", help="Generate summary CSV")
    parser.add_argument("--summary-csv", type=str, default=None)

    args = parser.parse_args()

    apply_flag = not args.no_apply
    weights_obj = json.loads(args.weights) if args.weights else None

    if getattr(args, "summary_score_speed", False):
        groups = [s.strip() for s in args.groups.split(",")] if args.groups else GROUP_TYPES_DEFAULT
        out_path, df_out = summarize_groups_score_speed_mix(
            groups,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            grid_step=args.grid_step,
            weights=weights_obj,
            output_csv=args.summary_csv,
            score_type=args.score_type,
        )
        print(out_path)
        print(df_out.to_string(index=False))
    
    elif args.folder:
        result = calibrate_subject_folder_mix(
            args.folder,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            grid_step=args.grid_step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.file:
        result = calibrate_file_mix(
            args.file,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            grid_step=args.grid_step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.groups:
        groups = [s.strip() for s in args.groups.split(",")]
        result = calibrate_groups_mix(
            groups,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            grid_step=args.grid_step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        result = calibrate_groups_mix(
            GROUP_TYPES_DEFAULT,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            grid_step=args.grid_step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))