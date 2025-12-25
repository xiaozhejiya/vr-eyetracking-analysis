import os
import re
import pandas as pd
import numpy as np

def project_root():
    """
    返回项目根目录：当前文件所在目录的上一级。
    用于统一拼接 data/MLP_data 等路径，避免硬编码绝对路径。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def ensure_dir(p):
    """
    确保目录 p 存在，如果不存在则递归创建。
    返回目录本身，方便链式调用。
    """
    os.makedirs(p, exist_ok=True)
    return p

def parse_q_num(name):
    """
    从字符串中解析题号 Qn 中的 n（例如 '...q3...' -> 3）。
    若未匹配到，则默认返回 1。
    """
    m = re.search(r"q(\d)", name.lower())
    return int(m.group(1)) if m else 1

def roi_cat(s):
    """
    根据 ROI 名称字符串归类到粗粒度类别：KW / INST / BG / OTHER / NONE。

    约定：
    - 以 "KW_" 开头视为关键词区域，归为 "KW"
    - 以 "INST_" 开头视为说明/指令区域，归为 "INST"
    - 以 "BG_" 开头视为背景区域，归为 "BG"
    - 其它非空字符串归为 "OTHER"
    - 非字符串（缺失值等）归为 "NONE"
    """
    if not isinstance(s, str):
        return "NONE"
    s = s.upper()
    if s.startswith("KW_"):
        return "KW"
    if s.startswith("INST_"):
        return "INST"
    if s.startswith("BG_"):
        return "BG"
    return "OTHER"

def aggregate_subject(df):
    """
    对“某一受试者 + 某一题目”的事件级数据（fixation/saccade 表）做特征汇总。

    输入
    ----
    df : pd.DataFrame
        至少包含列：["group", "subject", "ADQ_ID", "EventType", "Duration_ms",
                   "Amplitude_deg", "MaxVel", "MeanVel", "ROI"]

        约定：
        - EventType = "fixation" 或 "saccade"
        - ROI 为字符串标签，如 "KW_..." / "INST_..." / "BG_..." 等

    输出
    ----
    dict :
        聚合到“受试者-题目”粒度的一行特征，包括：
        - group / subject / q（题号）
        - fixation / saccade 的数量与总时长
        - fixation / saccade 的平均振幅（Amplitude_deg）
        - saccade 的 MaxVel / MeanVel 平均值
        - 落在 KW / INST / BG 区域的注视总时长（ms）
        - KW / INST / BG 在总时长中的占比（time ratio）
    """
    # 基本信息：受试者、组别、题号
    subj = df["subject"].iloc[0] if len(df) else ""
    grp = df["group"].iloc[0] if len(df) else ""
    q = df["ADQ_ID"].iloc[0] if len(df) else ""
    qn = parse_q_num(str(q))
    
    # 确保按时间顺序排列
    if "StartIndex" in df.columns:
        df = df.sort_values("StartIndex").reset_index(drop=True)

    # 映射 ROI 到粗类别（KW / INST / BG / OTHER / NONE）
    df["ROI_CAT"] = [roi_cat(x) for x in df["ROI"]]

    # 拆分为注视(fixation)与扫视(saccade)
    fx = df[df["EventType"] == "fixation"].copy()
    sc = df[df["EventType"] == "saccade"].copy()

    # --- 1. 基础统计 ---
    # 注视与扫视总时长（毫秒）
    fx_dur = float(fx["Duration_ms"].sum()) if len(fx) else 0.0
    sc_dur = float(sc["Duration_ms"].sum()) if len(sc) else 0.0
    total_dur = fx_dur + sc_dur  # 反应时/总时长

    # 注视与扫视计数
    fx_cnt = int(len(fx))
    sc_cnt = int(len(sc))

    # 注视/扫视振幅均值（度）
    fx_amp_mean = float(fx["Amplitude_deg"].mean()) if len(fx) else np.nan
    sc_amp_mean = float(sc["Amplitude_deg"].mean()) if len(sc) else np.nan

    # 扫视的最大速度与平均速度均值
    sc_maxvel_mean = float(sc["MaxVel"].mean()) if len(sc) else np.nan
    sc_meanvel_mean = float(sc["MeanVel"].mean()) if len(sc) else np.nan

    # --- 2. ROI 关注度 ---
    kw_fx_dur = float(fx[fx["ROI_CAT"] == "KW"]["Duration_ms"].sum()) if len(fx) else 0.0
    inst_fx_dur = float(fx[fx["ROI_CAT"] == "INST"]["Duration_ms"].sum()) if len(fx) else 0.0
    bg_fx_dur = float(fx[fx["ROI_CAT"] == "BG"]["Duration_ms"].sum()) if len(fx) else 0.0

    kw_ratio = (kw_fx_dur / total_dur) if total_dur > 0 else np.nan
    inst_ratio = (inst_fx_dur / total_dur) if total_dur > 0 else np.nan
    bg_ratio = (bg_fx_dur / total_dur) if total_dur > 0 else np.nan

    # --- 3. 首注视时间 (TTFF) ---
    # 找到第一个落在 KW 区域的 fixation
    ttff_ms = np.nan
    kw_indices = df[(df["EventType"] == "fixation") & (df["ROI_CAT"] == "KW")].index
    if len(kw_indices) > 0:
        first_kw_idx = kw_indices[0]
        # 计算该事件之前所有事件的总时长 (估算延迟)
        # 注意：这里假设 StartIndex 对应的相对时间可以用累积时长近似，
        # 或者更准确地，如果 StartIndex 是数据点索引，采样率约 60Hz (16.67ms/sample)
        # 但我们没有采样率信息，且可能有数据丢失/眨眼。
        # 最稳妥是用 StartIndex * 16.67ms (假设60Hz) 或者 累积 Duration
        # 这里使用累积 Duration 作为近似
        pre_events = df.iloc[:first_kw_idx]
        ttff_ms = float(pre_events["Duration_ms"].sum())
    
    # --- 4. ROI 转移 (Transitions) ---
    # 提取 fixation 的 ROI 序列 (忽略 None/Other 如果需要，这里保留)
    roi_seq = fx["ROI_CAT"].tolist()
    # 合并连续相同的 ROI (例如 KW, KW -> KW)
    collapsed_seq = []
    if roi_seq:
        collapsed_seq = [roi_seq[0]]
        for r in roi_seq[1:]:
            if r != collapsed_seq[-1]:
                collapsed_seq.append(r)
    
    # 统计转移
    trans_counts = {
        "KW_to_INST": 0, "INST_to_KW": 0,
        "KW_to_BG": 0, "BG_to_KW": 0,
        "INST_to_BG": 0, "BG_to_INST": 0
    }
    total_trans = 0
    revisit_kw_count = 0
    
    last_kw_index = -1
    kw_visit_intervals = []

    for i in range(len(collapsed_seq) - 1):
        src, dst = collapsed_seq[i], collapsed_seq[i+1]
        key = f"{src}_to_{dst}"
        if key in trans_counts:
            trans_counts[key] += 1
        total_trans += 1
        
        # Revisit KW: 任何非 KW -> KW 的转移
        if dst == "KW" and src != "KW":
            revisit_kw_count += 1
            
            # 计算间隔 (简单估算: 两个 visit 之间的索引差? 不，应该用时间)
            # 由于 collapsed_seq 丢失了时间信息，这里较难精确计算 interval
            # 我们改用原始 fx 列表来计算 interval
            pass

    # 计算 Revisit Interval (基于原始 fixation 列表)
    # 找到所有 KW fixation 的段
    kw_fix_segments = [] # [(start_time, end_time), ...]
    curr_start = 0.0
    for _, row in df.iterrows():
        dur = row["Duration_ms"]
        if row["EventType"] == "fixation" and row["ROI_CAT"] == "KW":
            kw_fix_segments.append((curr_start, curr_start + dur))
        curr_start += dur
    
    # 合并连续的时间段 (因为可能有连续的 KW fixations)
    merged_kw_seg = []
    if kw_fix_segments:
        curr_seg = kw_fix_segments[0]
        for next_seg in kw_fix_segments[1:]:
            # 如果中间间隔很小 (比如只隔了一个短 saccade?) 
            # 简单起见，只要是时间上断开就算一次新的 visit
            # 但我们只关心 "Revisit"，即中间去过别的地方
            # 上面的 collapsed_seq 已经定义了 "去过别的地方"
            pass
            
    # 简化版 Revisit Interval: (Total Duration - Total KW Duration) / Revisit Count
    # 这代表了 "平均每次离开 KW 后多久回来"
    avg_revisit_interval = np.nan
    if revisit_kw_count > 0:
        other_dur = total_dur - kw_fx_dur
        avg_revisit_interval = other_dur / revisit_kw_count

    # --- 5. 空间离散度与扫描路径 ---
    # Scanpath Length: 所有 Saccade 的幅度之和
    scanpath_length = float(sc["Amplitude_deg"].sum()) if len(sc) else 0.0
    
    # Fixation Dispersion: 注视点坐标的标准差 (如果有坐标)
    fix_dispersion = np.nan
    if "MeanX" in fx.columns and "MeanY" in fx.columns:
        # 移除 NaN
        valid_fx = fx.dropna(subset=["MeanX", "MeanY"])
        if len(valid_fx) > 1:
            std_x = valid_fx["MeanX"].std()
            std_y = valid_fx["MeanY"].std()
            # 综合离散度: sqrt(std_x^2 + std_y^2)
            fix_dispersion = np.sqrt(std_x**2 + std_y**2)
        elif len(valid_fx) == 1:
            fix_dispersion = 0.0

    out = {
        "group": grp,
        "subject": subj,
        "q": qn,
        # 基础
        "fixation_count": fx_cnt,
        "saccade_count": sc_cnt,
        "total_duration_ms": total_dur, # 反应时
        # 时长与占比
        "kw_fix_duration_ms": kw_fx_dur,
        "inst_fix_duration_ms": inst_fx_dur,
        "bg_fix_duration_ms": bg_fx_dur,
        "kw_time_ratio": kw_ratio,
        "inst_time_ratio": inst_ratio,
        "bg_time_ratio": bg_ratio,
        # 新增指标
        "ttff_ms": ttff_ms, # 首注视时间
        "scanpath_length": scanpath_length, # 扫描路径长
        "fixation_dispersion": fix_dispersion, # 空间离散度
        "revisit_kw_count": revisit_kw_count, # 回视次数
        "avg_revisit_interval": avg_revisit_interval, # 回视间隔
        "roi_trans_total": total_trans, # 总转移次数
        # 细分转移
        "trans_kw_inst": trans_counts["KW_to_INST"] + trans_counts["INST_to_KW"],
        "trans_kw_bg": trans_counts["KW_to_BG"] + trans_counts["BG_to_KW"],
        "trans_inst_bg": trans_counts["INST_to_BG"] + trans_counts["BG_to_INST"],
        # 原始均值
        "fixation_amplitude_mean_deg": fx_amp_mean,
        "saccade_amplitude_mean_deg": sc_amp_mean,
        "saccade_maxvel_mean": sc_maxvel_mean,
        "saccade_meanvel_mean": sc_meanvel_mean,
    }
    return out

def aggregate_file(fp):
    """
    对某个题目的“事件级数据文件”做汇总。
    该文件一般由 export_group_events 生成，如：control_q1.csv，
    内部包含多个受试者的事件行。

    处理逻辑：
      - 若文件不存在或为空，返回空 DataFrame
      - 否则按 subject 分组，对每个受试者调用 aggregate_subject，
        最终返回 “每个受试者一行特征” 的 DataFrame。
    """
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp)
    if df.empty:
        return pd.DataFrame()

    # 按 subject 聚合，每个受试者做一次特征计算
    g = df.groupby("subject")
    rows = [aggregate_subject(gdf.copy()) for _, gdf in g]
    return pd.DataFrame(rows)

def aggregate_for_group(source_root, group):
    """
    针对单个组别（如 control/mci/ad），对 Q1–Q5 的事件数据做特征汇总。

    输入
    ----
    source_root : str
        事件级数据根目录，形如 data/MLP_data/event_data。
        其下结构为：{group}_group/{group}_q{q}.csv
    group : str
        组别名称，如 "control"。

    输出
    ----
    dict :
        key 为题号 q（1–5），value 为写出的特征 CSV 路径。
        特征文件路径形如：
            data/MLP_data/features/event_features/{group}_group/{group}_q{q}.csv
    """
    # 输出根目录：按组别单独建一个 event_features/{group}_group 子目录
    out_root = ensure_dir(
        os.path.join(
            project_root(),
            "data",
            "MLP_data",
            "features",
            "event_features",
            f"{group}_group",
        )
    )

    saved = {}
    for q in [1, 2, 3, 4, 5]:
        # 原始事件数据文件（由 export_group_events 生成）
        fp = os.path.join(source_root, f"{group}_group", f"{group}_q{q}.csv")
        # 对该题目做 subject 粒度的汇总
        df_out = aggregate_file(fp)
        # 写出特征文件
        out_fp = os.path.join(out_root, f"{group}_q{q}.csv")
        df_out.to_csv(out_fp, index=False)
        saved[q] = out_fp

    return saved

def aggregate_all(groups):
    """
    对多个组别（例如 ["control","mci","ad"]）批量执行特征汇总。

    输入
    ----
    groups : List[str]
        需要处理的组别列表。

    输出
    ----
    dict :
        最外层 key 为 group 名，内层为 {q: feature_csv_path} 的字典。
    """
    # 事件级数据统一存放的位置
    source_root = os.path.join(project_root(), "data", "MLP_data", "event_data")
    res = {}
    for g in groups:
        res[g] = aggregate_for_group(source_root, g)
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Aggregate event-level eye-tracking features into subject-level features per group/question."
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="control,mci,ad",
        help="逗号分隔的组别列表，例如：control,mci,ad",
    )
    args = parser.parse_args()

    # 解析组别列表，去掉空白项
    groups = [s.strip() for s in args.groups.split(",") if s.strip()]

    # 对所有指定组别执行特征汇总
    result = aggregate_all(groups)
    print(result)
