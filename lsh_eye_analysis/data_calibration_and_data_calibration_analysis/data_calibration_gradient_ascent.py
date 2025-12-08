import os
import re
import sys
import numpy as np
import pandas as pd
import torch

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
from lsh_eye_analysis.utils.score_function import calculate_score_and_metrics, apply_offset


sys.path.append(project_root())
from lsh_eye_analysis.utils.score_function import get_dt


def import_event_analyzer():
    """
    动态导入 EventAnalyzer。
    之所以不在文件顶部直接 import，是为了在作为脚本执行和被其它模块 import 时都能正常找到项目根路径。
    """
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()


def calibration_output_dir():
    """
    所有校准后 CSV 的统一输出根目录。
    下面还会按 group / subject 再分子目录。
    """
    return os.path.join(project_root(), "lsh_eye_analysis", "data", "data_calibration_gradient_ascent")


def calibration_output_path(file_path):
    """
    根据原始文件路径构建校准结果的输出路径：

    - 原文件名 *_preprocessed.csv -> *_preprocessed_calibrated.csv
    - 原文件位于 data/{group}_processed/{subject}/xxx.csv
      -> 输出到 lsh_eye_analysis/data_calibration_gradient_ascent/{group}_calibrated/{subject}/xxx_calibrated.csv

    如果找不到 group/subject 结构，则直接输出到 calibration_output_dir() 下面。
    """
    base = os.path.basename(file_path).replace(
        "_preprocessed.csv", "_preprocessed_calibrated.csv"
    )

    data_root = data_dir()
    try:
        # 计算 file_path 相对于 data_root 的相对路径
        rel = os.path.relpath(file_path, data_root)
    except ValueError:
        # 不在 data_root 之下
        rel = None

    group = None
    subject = None
    if rel:
        parts = rel.split(os.sep)
        # 形如 control_processed/subject_001/xxx.csv
        if len(parts) >= 2 and parts[0].endswith("_processed"):
            group = parts[0].replace("_processed", "")
            subject = parts[1]

    # 根据 group/subject 组建输出目录
    if group:
        out_dir = os.path.join(calibration_output_dir(), f"{group}_calibrated")
        if subject:
            out_dir = os.path.join(out_dir, subject)
    else:
        out_dir = calibration_output_dir()

    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


def data_dir(*parts):
    """
    默认数据根目录改为 lsh_eye_analysis/data/data_processing。
    例如：
        data_dir('control_processed') -> <project_root>/lsh_eye_analysis/data/data_processed/control_processed

    参数 *parts 可以是多级子目录。
    """
    return os.path.join(project_root(), "lsh_eye_analysis", "data", "data_processed", *parts)


def group_root(group):
    """
    某个分组数据根目录，例如 group='control' 时：
        -> data/control_processed
    """
    return data_dir(f"{group}_processed")


def list_subject_folders(group):
    """
    列出某个分组下所有被试目录（仅目录，不含文件）。
    """
    root = group_root(group)
    if not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def calibrate_groups(
    groups,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
    apply=False,
    score_type="soft",
):
    """
    对若干 group 进行批量校准。

    对于每个 group：
      - 找到它下面所有 subject folder
      - 对每个 subject folder 调用 calibrate_subject_folder

    参数
    ----
    groups : list[str]
        需要处理的 group 名称列表，例如 ["control", "mci", "ad"]。
    dx_bounds, dy_bounds : (float, float)
        平移搜索范围 [min, max]，单位是归一化坐标。
    step : float
        网格搜索的步长。
    weights : dict 或 None
        打分权重，传给 optimize_offset_by_roi。
    apply : bool
        是否将校准结果写回 CSV 文件。
    """
    results = []
    for g in groups:
        folders = list_subject_folders(g)
        for folder in folders:
            res = calibrate_subject_folder(
                folder,
                dx_bounds=dx_bounds,
                dy_bounds=dy_bounds,
                step=step,
                weights=weights,
                apply=apply,
                score_type=score_type,
            )
            results.append({"group": g, "folder": folder, "results": res})
    return results


def calibrate_groups_under_dir(
    parent_dir,
    groups=None,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
    apply=False,
    score_type="soft",
):
    if groups is None:
        groups = GROUP_TYPES_DEFAULT
    results = []
    for g in groups:
        group_path = os.path.join(parent_dir, f"{g}_processed")
        if not os.path.exists(group_path):
            continue
        for d in os.listdir(group_path):
            folder_path = os.path.join(group_path, d)
            if not os.path.isdir(folder_path):
                continue
            res = calibrate_subject_folder(
                folder_path,
                dx_bounds=dx_bounds,
                dy_bounds=dy_bounds,
                step=step,
                weights=weights,
                apply=apply,
                score_type=score_type,
            )
            results.append({"group": g, "folder": folder_path, "results": res})
    return results


def parse_q_num_from_filename(filename):
    """
    从文件名中解析题号，例如 'xxx_q2_preprocessed.csv' -> 2。

    正则：r"q(\\d)"，只匹配一位数字。
    找不到时默认返回 1。
    """
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1


def optimize_offset_by_roi(
    df,
    roi_kw,
    roi_inst,
    roi_bg,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,          # 保留参数，但梯度版不用 step
    weights=None,
    score_type="soft",
):
    """
    使用 soft ROI + 梯度上升，在连续的 (dx, dy) 空间搜索最优平移，
    目标仍然是“inst/kw 时间多、背景时间少”。

    背景 bg 在这里不再直接用 roi_bg，而是定义为 inst/kw 的补集：
        p_bg(x,y) = 1 - clip(p_inst(x,y) + p_kw(x,y), 0, 1)

    评分函数（soft 版）：
        T_inst = sum_i dt_i * p_inst(x_i+dx, y_i+dy)
        T_kw   = sum_i dt_i * p_kw(  x_i+dx, y_i+dy)
        T_bg   = sum_i dt_i * p_bg(  x_i+dx, y_i+dy)

        num   = w_inst * T_inst + w_kw * T_kw
        denom = num + w_bg * T_bg

        time_ratio = num / (denom + eps)

        score(dx,dy) = time_ratio - lambda_reg * (dx^2 + dy^2)

    返回值结构与原先保持一致。
    """
    # -----------------------------
    # 0. 基本检查和权重处理
    # -----------------------------
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}

    if weights is None:
        weights = {
            "inst_time": 1.0,
            "kw_time": 1.0,
            "bg_time": 0.5,
            # 正则强度（可选），控制位移不要过大
            "lambda_reg": 0.0,
        }

    w_inst = float(weights.get("inst_time", 1.0))
    w_kw   = float(weights.get("kw_time", 1.0))
    w_bg   = float(weights.get("bg_time", 0.5))
    lambda_reg = float(weights.get("lambda_reg", 0.0))

    # 原始 numpy 数据
    xs_np = df["x"].to_numpy(dtype=float)
    ys_np = df["y"].to_numpy(dtype=float)
    dt_np = get_dt(df)

    # -----------------------------
    # 1. 转成 torch 张量
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(xs_np, dtype=torch.float32, device=device)
    y = torch.tensor(ys_np, dtype=torch.float32, device=device)
    dt = torch.tensor(dt_np, dtype=torch.float32, device=device)

    dx = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    dy = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))

    # 优化器 & 超参数：可以视情况调 lr/n_steps
    lr = 0.05
    n_steps = 200
    optimizer = torch.optim.Adam([dx, dy], lr=lr)

    # -----------------------------
    # 2. soft ROI membership 函数
    # -----------------------------
    def soft_prob(xs, ys, roi_list, k=35.0):
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

    best_score = None
    best_dx = 0.0
    best_dy = 0.0

    # -----------------------------
    # 3. 梯度上升迭代
    # -----------------------------
    for step_idx in range(n_steps):
        optimizer.zero_grad()

        # 平移 + 裁剪到 [0,1]
        xs = torch.clamp(x + dx, 0.0, 1.0)
        ys = torch.clamp(y + dy, 0.0, 1.0)

        # soft 概率：inst / kw
        p_inst = soft_prob(xs, ys, roi_inst)
        p_kw   = soft_prob(xs, ys, roi_kw)

        # 背景 = inst/kw 的补集
        p_pos = torch.clamp(p_inst + p_kw, 0.0, 1.0)
        p_bg  = 1.0 - p_pos

        # soft 时间
        T_inst = (dt * p_inst).sum()
        T_kw   = (dt * p_kw).sum()
        T_bg   = (dt * p_bg).sum()

        num   = w_inst * T_inst + w_kw * T_kw
        denom = num + w_bg * T_bg + 1e-6  # 防止除 0

        time_ratio = num / denom

        # L2 正则，避免 dx,dy 飘太远
        reg = lambda_reg * (dx * dx + dy * dy)

        score = time_ratio - reg

        # 梯度上升 → 最小化 -score
        loss = -score
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            s_val = score.item()
            if (best_score is None) or (s_val > best_score):
                best_score = s_val
                best_dx = dx.item()
                best_dy = dy.item()

    # -----------------------------
    # 4. 使用最优 (dx,dy) 回到 numpy，用指定的 score_type 算 metrics
    # -----------------------------
    
    score, metrics = calculate_score_and_metrics(
        df, float(best_dx), float(best_dy), roi_kw, roi_inst, weights, score_type=score_type
    )

    best = {
        "dx": float(best_dx),
        "dy": float(best_dy),
        "score": float(score),   # 使用 calculate_score_and_metrics 返回的 score
        "metrics": metrics,
    }

    return best


def calibrate_file_by_roi_grid(
    file_path,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
    apply=False,
    save_path=None,
    score_type="soft",
):
    """
    对单个 *_preprocessed.csv 文件进行基于 ROI 的网格平移校准。

    步骤：
      1) 读入 CSV -> DataFrame
      2) 根据文件名解析题号 q
      3) 用 EventAnalyzer.get_roi_def("n2q{q}") 拿到 ROI 定义
      4) 调用 optimize_offset_by_roi 搜索最优 (dx, dy)
      5) 若 apply=True，则将校准后坐标写回新的 CSV 文件

    返回
    ----
    dict:
      {
        "file_path": 原始文件路径,
        "q":         题号,
        "best":      optimize_offset_by_roi 的返回结果,
        "applied_path": 实际写出的 CSV 路径（若 apply=False 则为 None）
      }
    """
    df = pd.read_csv(file_path)
    q = parse_q_num_from_filename(os.path.basename(file_path))

    analyzer = import_event_analyzer()
    # kw: 关键词区域；inst: 指令区域；bg: 背景区域
    kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")

    best = optimize_offset_by_roi(
        df,
        kw,
        inst,
        bg,
        dx_bounds=dx_bounds,
        dy_bounds=dy_bounds,
        step=step,
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


def calibrate_subject_folder(
    folder,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
    apply=False,
    score_type="soft",
):
    """
    对某一被试目录（subject folder）下的所有 *_preprocessed.csv 文件逐个进行校准。

    注意：当前实现是“每道题单独搜索一对 (dx, dy)”。
    如果以后要改成“一个被试统一一对 (dx, dy)”，可以在这里聚合数据后统一调用新的优化函数。

    返回
    ----
    list[dict] : 每个文件 calibrate_file_by_roi_grid 的结果。
    """
    results = []
    for name in os.listdir(folder):
        if name.endswith("_preprocessed.csv"):
            fp = os.path.join(folder, name)
            res = calibrate_file_by_roi_grid(
                fp,
                dx_bounds=dx_bounds,
                dy_bounds=dy_bounds,
                step=step,
                weights=weights,
                apply=apply,
                score_type=score_type,
            )
            results.append(res)
    return results


def summarize_groups_score_speed(
    groups,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
    output_csv=None,
):
    import time
    analyzer = import_event_analyzer()
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
                    res = calibrate_file_by_roi_grid(
                        fp,
                        dx_bounds=dx_bounds,
                        dy_bounds=dy_bounds,
                        step=step,
                        weights=weights,
                        apply=False,
                    )
                    t1 = time.perf_counter()
                    proc_ms_q[q].append(float((t1 - t0) * 1000.0))
                    if res and res.get("best"):
                        scores_q[q].append(float(res["best"]["score"]))
        row = {
            "group": g,
            "avg_score_q1": float(np.mean(scores_q[1])) if len(scores_q[1]) else float("nan"),
            "avg_score_q2": float(np.mean(scores_q[2])) if len(scores_q[2]) else float("nan"),
            "avg_score_q3": float(np.mean(scores_q[3])) if len(scores_q[3]) else float("nan"),
            "avg_score_q4": float(np.mean(scores_q[4])) if len(scores_q[4]) else float("nan"),
            "avg_score_q5": float(np.mean(scores_q[5])) if len(scores_q[5]) else float("nan"),
            "avg_proc_ms_q1": float(np.mean(proc_ms_q[1])) if len(proc_ms_q[1]) else float("nan"),
            "avg_proc_ms_q2": float(np.mean(proc_ms_q[2])) if len(proc_ms_q[2]) else float("nan"),
            "avg_proc_ms_q3": float(np.mean(proc_ms_q[3])) if len(proc_ms_q[3]) else float("nan"),
            "avg_proc_ms_q4": float(np.mean(proc_ms_q[4])) if len(proc_ms_q[4]) else float("nan"),
            "avg_proc_ms_q5": float(np.mean(proc_ms_q[5])) if len(proc_ms_q[5]) else float("nan"),
        }
        rows.append(row)
    df_out = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = os.path.join(calibration_output_dir(), "score_proc_summary.csv")
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_out.to_csv(output_csv, index=False)
    except Exception:
        pass
    return output_csv, df_out


if __name__ == "__main__":
    import argparse
    import json

    # 命令行接口，方便批量处理
    parser = argparse.ArgumentParser(description="ROI-driven grid calibration")
    parser.add_argument(
        "--file", type=str, help="path to a single _preprocessed.csv file"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="path to a subject folder containing _preprocessed.csv files",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="comma-separated group types, e.g. control,sci,ad",
    )
    parser.add_argument(
        "--groups-parent",
        type=str,
        default=None,
        help="path to parent dir containing <group>_processed subdirs",
    )
    # 默认平移范围设置得比较大，以覆盖较大的系统偏移
    parser.add_argument("--dx-min", type=float, default=-0.25)
    parser.add_argument("--dx-max", type=float, default=0.25)
    parser.add_argument("--dy-min", type=float, default=-0.25)
    parser.add_argument("--dy-max", type=float, default=0.25)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help='JSON string of weights, e.g. {"inst_time":1.2,"bg_time":0.8}',
    )
    parser.add_argument(
        "--no-apply", action="store_true", help="do not write calibrated CSVs"
    )
    parser.add_argument(
        "--summary-score-speed", action="store_true",
        help="summarize per-group per-question (Q1–Q5) average score and processing time (ms) and write CSV"
    )
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["hard", "soft"],
        default="soft",
        help="Score calculation type: 'hard' (rectangular) or 'soft' (sigmoid heatmap). Default: soft"
    )
    parser.add_argument(
        "--summary-csv", type=str, default=None,
        help="output CSV path for summary (default: lsh_eye_analysis/data_calibration_gradient_ascent/score_speed_summary.csv)"
    )
    args = parser.parse_args()

    # 是否写入校准结果
    apply_flag = not args.no_apply

    # 解析权重 JSON
    weights_obj = None
    if args.weights:
        try:
            weights_obj = json.loads(args.weights)
        except Exception:
            weights_obj = None

    # 优先级：
    #   1) 若指定 --groups-parent，则在该目录下按 groups 处理
    #   2) 若指定 --folder，则对一个 subject folder 进行处理
    #   3) 若指定 --file，则只处理一个文件
    #   4) 若指定 --groups，则对多个 group 批量处理
    #   5) 否则，对 GROUP_TYPES_DEFAULT 中所有 group 进行处理
    if getattr(args, "summary_score_speed", False):
        groups = [s.strip() for s in args.groups.split(",") if s.strip()] if args.groups else GROUP_TYPES_DEFAULT
        out_path, df_out = summarize_groups_score_speed(
            groups,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            output_csv=args.summary_csv,
        )
        print(out_path)
        print(df_out.to_string(index=False))
    elif args.groups_parent:
        groups = [s.strip() for s in args.groups.split(",") if s.strip()] if args.groups else GROUP_TYPES_DEFAULT
        result = calibrate_groups_under_dir(
            args.groups_parent,
            groups=groups,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.folder:
        result = calibrate_subject_folder(
            args.folder,
            dx_bounds=(args.dx_min, args.dx_max),
            dy_bounds=(args.dy_min, args.dy_max),
            step=args.step,
            weights=weights_obj,
            apply=apply_flag,
            score_type=args.score_type,
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
            score_type=args.score_type,
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
            score_type=args.score_type,
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
            score_type=args.score_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
