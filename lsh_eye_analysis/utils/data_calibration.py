import os
import re
import sys
import numpy as np
import pandas as pd

# 默认包含的被试分组名称（和 data/ 下面的目录对应）
GROUP_TYPES_DEFAULT = ["control", "mci", "ad"]


def project_root():
    """
    返回项目根目录：
    当前文件 -> 上级目录 -> 上上级目录
    data/, analysis/ 等目录都假定挂在这个根目录下。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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
    return os.path.join(project_root(), "lsh_eye_analysis", "data_calibartion")


def calibration_output_path(file_path):
    """
    根据原始文件路径构建校准结果的输出路径：

    - 原文件名 *_preprocessed.csv -> *_preprocessed_calibrated.csv
    - 原文件位于 data/{group}_processed/{subject}/xxx.csv
      -> 输出到 lsh_eye_analysis/data_calibartion/{group}_calibrated/{subject}/xxx_calibrated.csv

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
    拼接 data/ 子目录的路径，例如：
        data_dir('control_processed') -> <project_root>/data/control_processed

    参数 *parts 可以是多级子目录。
    """
    return os.path.join(project_root(), "data", *parts)


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
            )
            results.append({"group": g, "folder": folder, "results": res})
    return results


def parse_q_num_from_filename(filename):
    """
    从文件名中解析题号，例如 'xxx_q2_preprocessed.csv' -> 2。

    正则：r"q(\\d)"，只匹配一位数字。
    找不到时默认返回 1。
    """
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1


def _get_dt(df):
    """
    计算每一行样本对应的时间间隔 dt[i]（单位与原始列一致，通常是毫秒）。

    1) 若存在 `time_diff` 列，直接使用：
        dt[i] = time_diff[i]
       并将 NaN 替换为 0。

    2) 若存在 `milliseconds` 列（绝对时间戳），使用一阶差分：
        d[i] = ms[i] - ms[i-1],  i >= 1
        d[0] = ms[0] - ms[0] = 0
       然后将负值（由于重置/回绕）强制置为 0：
        dt[i] = max(d[i], 0)

    3) 否则，认为每一行 dt = 1（等间隔采样）。
    """
    if "time_diff" in df.columns:
        dt = df["time_diff"].to_numpy()
        return np.where(np.isnan(dt), 0.0, dt)

    if "milliseconds" in df.columns:
        ms = df["milliseconds"].to_numpy()
        d = np.diff(ms, prepend=ms[0])
        return np.where(d < 0, 0.0, d)

    # 兜底：假设每一帧时间相同
    return np.ones(len(df), dtype=float)


def _in_any_roi(xs, ys, roi_list):
    """
    判断每个 (x, y) 是否落在给定 roi_list 的任意一个矩形 ROI 中。

    roi_list 的元素格式：
        (name, x_min, y_min, x_max, y_max)
    坐标假定已经归一化到 [0,1] 范围。

    返回值：
        inside : bool 数组，inside[i] 为 True 表示第 i 个点落在至少一个 ROI 内。
    """
    if not roi_list:
        return np.zeros_like(xs, dtype=bool)

    inside = np.zeros_like(xs, dtype=bool)
    for _, xmn, ymn, xmx, ymy in roi_list:
        # 条件：
        #   x_min <= x <= x_max
        #   y_min <= y <= y_max
        inside |= (xs >= xmn) & (xs <= xmx) & (ys >= ymn) & (ys <= ymy)
    return inside


def _enter_count(mask):
    """
    统计布尔序列 mask 中从 False -> True 的“进入次数”。

    设 mask[i] 表示第 i 帧是否在某个区域内，
    我们统计：
        enter_count = |{ i | mask[i-1] == False, mask[i] == True, i >= 1 }|

    参数
    ----
    mask : np.ndarray[bool]

    返回
    ----
    int : 进入次数
    """
    if len(mask) < 2:
        return 0
    prev = mask[:-1]
    curr = mask[1:]
    return int(np.sum((~prev) & curr))


def apply_offset(df, dx, dy):
    """
    将 (x, y) 坐标整体平移 (dx, dy)，并裁剪到 [0,1] 范围。

    变换公式：
        x' = clip(x + dx, 0, 1)
        y' = clip(y + dy, 0, 1)
    """
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


def optimize_offset_by_roi(
    df,
    roi_kw,
    roi_inst,
    roi_bg,
    dx_bounds=(-0.05, 0.05),
    dy_bounds=(-0.05, 0.05),
    step=0.005,
    weights=None,
):
    """
    在给定的 dx, dy 网格上搜索最优平移，使得“看指令 / 关键词时间多、看背景时间少”。

    参数
    ----
    df : pd.DataFrame
        包含 x, y 和时间信息（time_diff 或 milliseconds）的眼动数据。
    roi_kw : list
        关键词区域 ROI 列表。
    roi_inst : list
        指令区域 ROI 列表。
    roi_bg : list
        背景区域 ROI 列表。
    dx_bounds, dy_bounds : (float, float)
        搜索的平移范围 [min, max]。
    step : float
        网格步长。
    weights : dict 或 None
        各种指标的权重。如果为 None，则使用默认值：
            inst_time: 1.0   # 指令区域停留时间
            kw_time:   1.0   # 关键词区域停留时间
            bg_time:   0.5   # 背景区域停留时间（惩罚项）
            inst_enter, kw_enter, bg_enter : 进入次数的权重（默认 0）

    目标函数（score）的形式：

        记
            T_inst(dx, dy) = 指令 ROI 内的总时间
            T_kw(dx, dy)   = 关键词 ROI 内的总时间
            T_bg(dx, dy)   = 背景 ROI 内的总时间

            E_inst(dx, dy) = 指令 ROI 的进入次数
            E_kw(dx, dy)   = 关键词 ROI 的进入次数
            E_bg(dx, dy)   = 背景 ROI 的进入次数

        则：
            score(dx, dy)
              = w_inst_time * T_inst
              + w_kw_time   * T_kw
              - w_bg_time   * T_bg
              + w_inst_enter * E_inst
              + w_kw_enter   * E_kw
              - w_bg_enter   * E_bg

        其中 w_* 由 weights 控制。

    返回
    ----
    dict:
        {
          "dx":   最优平移量 dx,
          "dy":   最优平移量 dy,
          "score": 最优得分,
          "metrics": {
              "inst_time": T_inst,
              "kw_time":   T_kw,
              "bg_time":   T_bg,
              "inst_enter":E_inst,
              "kw_enter":  E_kw,
              "bg_enter":  E_bg,
          }
        }
    """
    # -----------------------------
    # 1. 处理权重，构造权重向量 w_vec
    # -----------------------------
    if weights is None:
        weights = {
            "inst_time": 1.0,
            "kw_time": 1.0,
            "bg_time": 0.5,
            "inst_enter": 0.0,
            "kw_enter": 0.0,
            "bg_enter": 0.0,
        }

    # 按顺序把 6 个特征的权重排成向量：
    #   [w_inst_time, w_kw_time, -w_bg_time, w_inst_enter, w_kw_enter, -w_bg_enter]
    # 注意：对 bg_time / bg_enter 带负号，是因为在 score 中是惩罚项（要减去）。
    # 形状为 (1, 6)，后面便于做广播矩阵乘法。
    w_vec = np.array(
        [
            weights["inst_time"],
            weights["kw_time"],
            -weights["bg_time"],   # 惩罚背景时间
            weights["inst_enter"],
            weights["kw_enter"],
            -weights["bg_enter"],  # 惩罚背景进入次数
        ],
        dtype=float,
    ).reshape(1, 6)

    # df 中没有 x/y 或者为空，直接返回一个“无效”结果
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}

    # 原始眼动坐标与时间间隔
    xs0 = df["x"].to_numpy(dtype=float)  # 形状 (N,)
    ys0 = df["y"].to_numpy(dtype=float)  # 形状 (N,)
    dt = _get_dt(df)                     # 形状 (N,)

    # 构造 dx / dy 网格
    # dx_vals: (Dx,), dy_vals: (Dy,)
    dx_vals = np.arange(dx_bounds[0], dx_bounds[1] + 1e-12, step)
    dy_vals = np.arange(dy_bounds[0], dy_bounds[1] + 1e-12, step)

    # -------------------------------------------------------
    # 2. 一次性构造所有 (dx, dy) 下平移后的坐标张量 xs_adj / ys_adj
    # -------------------------------------------------------
    # xs_adj 的形状为 (N, Dx, Dy)：
    #   - 第 0 维：时间序列上的采样点 i = 0..N-1
    #   - 第 1 维：不同的 dx 值
    #   - 第 2 维：不同的 dy 值
    #
    # 公式：xs_adj[i, j, k] = clip(xs0[i] + dx_vals[j], 0, 1)
    xs_adj = np.clip(xs0[:, None, None] + dx_vals[None, :, None], 0.0, 1.0)

    # ys_adj 的形状同样为 (N, Dx, Dy)：
    #   ys_adj[i, j, k] = clip(ys0[i] + dy_vals[k], 0, 1)
    ys_adj = np.clip(ys0[:, None, None] + dy_vals[None, None, :], 0.0, 1.0)

    # -------------------------------------------------------
    # 3. 根据 ROI 构建 3D 布尔掩码：m_kw / m_inst / m_bg
    # -------------------------------------------------------
    def build_mask(roi_list):
        """
        为一个 ROI 列表构建 mask，返回形状 (N, Dx, Dy) 的布尔数组：
            m[i, j, k] = True 表示在第 i 个采样点、
                               平移 (dx_vals[j], dy_vals[k]) 下，
                               眼动落在 roi_list 任一矩形内。
        """
        if not roi_list:
            # 如果没有 ROI，直接返回全 False
            return np.zeros(
                (xs_adj.shape[0], xs_adj.shape[1], ys_adj.shape[2]), dtype=bool
            )

        # 初始化全 False 的 mask
        m = np.zeros(
            (xs_adj.shape[0], xs_adj.shape[1], ys_adj.shape[2]), dtype=bool
        )
        # 对每一个矩形 ROI 累积 “在此 ROI 内” 的条件
        for _, xmn, ymn, xmx, ymy in roi_list:
            # 对应条件：
            #   xmn <= x <= xmx 且 ymn <= y <= ymy
            m |= (xs_adj >= xmn) & (xs_adj <= xmx) & (ys_adj >= ymn) & (ys_adj <= ymy)
        return m

    # 关键词、指令、背景的掩码，形状均为 (N, Dx, Dy)
    m_kw = build_mask(roi_kw)
    m_inst = build_mask(roi_inst)
    m_bg = build_mask(roi_bg)

    # -------------------------------------------------------
    # 4. 计算每个 (dx, dy) 下的时间特征 T_* 和进入次数特征 E_*
    # -------------------------------------------------------
    # 为了和 m_* 广播相乘，把 dt 从 (N,) 扩展到 (N, 1, 1)
    # 这样 dt3[i, 0, 0] = dt[i]，广播到所有 (Dx, Dy)
    dt3 = dt[:, None, None]

    # 区域内时间：
    #   t_kw[j, k]   = sum_i dt[i] * 1{(i,j,k) 落在 KW ROI 内}
    #   t_inst[j, k] = sum_i dt[i] * 1{(i,j,k) 落在 INST ROI 内}
    #   t_bg[j, k]   = sum_i dt[i] * 1{(i,j,k) 落在 BG ROI 内}
    #
    # 计算顺序：
    #   (dt3 * m_kw) 形状为 (N, Dx, Dy)，沿 axis=0 求和 -> (Dx, Dy)
    t_kw = (dt3 * m_kw).sum(axis=0).astype(float)
    t_inst = (dt3 * m_inst).sum(axis=0).astype(float)
    t_bg = (dt3 * m_bg).sum(axis=0).astype(float)

    # 进入次数（E_*）：
    # 对时间维（第 0 维）做布尔差分，统计从 False -> True 的次数。
    # 例如 E_kw[j, k] = |{i >= 1 | ~m_kw[i-1, j, k] & m_kw[i, j, k]}|
    e_kw = ((~m_kw[:-1, :, :]) & m_kw[1:, :, :]).sum(axis=0)
    e_inst = ((~m_inst[:-1, :, :]) & m_inst[1:, :, :]).sum(axis=0)
    e_bg = ((~m_bg[:-1, :, :]) & m_bg[1:, :, :]).sum(axis=0)

    # -------------------------------------------------------
    # 5. 把所有特征堆叠成 feats，并用权重 w_vec 计算 score 矩阵
    # -------------------------------------------------------
    # feats 的形状为 (6, Dx, Dy)，依次存放：
    #   0: T_inst
    #   1: T_kw
    #   2: T_bg
    #   3: E_inst
    #   4: E_kw
    #   5: E_bg
    feats = np.stack(
        [
            t_inst,
            t_kw,
            t_bg,
            e_inst.astype(float),
            e_kw.astype(float),
            e_bg.astype(float),
        ],
        axis=0,
    )

    # 将权重向量 reshape 为 (6, 1, 1)，便于与 feats 做逐元素乘法：
    #   score_mat[j, k] = sum_{c=0..5} w_b[c,0,0] * feats[c, j, k]
    w_b = w_vec.reshape(6, 1, 1)
    score_mat = (w_b * feats).sum(axis=0)  # 形状为 (Dx, Dy)

    # -------------------------------------------------------
    # 6. 找到得分最高的 (dx_idx, dy_idx)，并整理最优结果
    # -------------------------------------------------------
    # idx 是一个二元 tuple：(dx_index, dy_index)
    idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)

    best = {
        # 最优平移量
        "dx": float(dx_vals[idx[0]]),
        "dy": float(dy_vals[idx[1]]),
        # 最优得分
        "score": float(score_mat[idx]),
        # 对应的各项指标
        "metrics": {
            "inst_time": float(t_inst[idx]),
            "kw_time": float(t_kw[idx]),
            "bg_time": float(t_bg[idx]),
            "inst_enter": int(e_inst[idx]),
            "kw_enter": int(e_kw[idx]),
            "bg_enter": int(e_bg[idx]),
        },
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
            )
            results.append(res)
    return results


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
    #   1) 若指定 --folder，则对一个 subject folder 进行处理
    #   2) 若指定 --file，则只处理一个文件
    #   3) 若指定 --groups，则对多个 group 批量处理
    #   4) 否则，对 GROUP_TYPES_DEFAULT 中所有 group 进行处理
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
