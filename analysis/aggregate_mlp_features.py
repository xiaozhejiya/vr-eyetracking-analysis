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

    # 映射 ROI 到粗类别（KW / INST / BG / OTHER / NONE）
    df["ROI_CAT"] = [roi_cat(x) for x in df["ROI"]]

    # 拆分为注视(fixation)与扫视(saccade)
    fx = df[df["EventType"] == "fixation"].copy()
    sc = df[df["EventType"] == "saccade"].copy()

    # 注视与扫视总时长（毫秒）
    fx_dur = float(fx["Duration_ms"].sum()) if len(fx) else 0.0
    sc_dur = float(sc["Duration_ms"].sum()) if len(sc) else 0.0

    # 注视与扫视计数
    fx_cnt = int(len(fx))
    sc_cnt = int(len(sc))

    # 注视/扫视振幅均值（度），若无对应事件则为 NaN
    fx_amp_mean = float(fx["Amplitude_deg"].mean()) if len(fx) else np.nan
    sc_amp_mean = float(sc["Amplitude_deg"].mean()) if len(sc) else np.nan

    # 扫视的最大速度与平均速度均值
    sc_maxvel_mean = float(sc["MaxVel"].mean()) if len(sc) else np.nan
    sc_meanvel_mean = float(sc["MeanVel"].mean()) if len(sc) else np.nan

    # 不同 ROI 区域内的注视总时长（只统计 fixation）
    kw_fx_dur = float(fx[fx["ROI_CAT"] == "KW"]["Duration_ms"].sum()) if len(fx) else 0.0
    inst_fx_dur = float(fx[fx["ROI_CAT"] == "INST"]["Duration_ms"].sum()) if len(fx) else 0.0
    bg_fx_dur = float(fx[fx["ROI_CAT"] == "BG"]["Duration_ms"].sum()) if len(fx) else 0.0

    # 总时长 = 注视时长 + 扫视时长
    total_dur = fx_dur + sc_dur

    # 不同 ROI 在总时长中的占比（time ratio）
    kw_ratio = (kw_fx_dur / total_dur) if total_dur > 0 else np.nan
    inst_ratio = (inst_fx_dur / total_dur) if total_dur > 0 else np.nan
    bg_ratio = (bg_fx_dur / total_dur) if total_dur > 0 else np.nan

    out = {
        "group": grp,
        "subject": subj,
        "q": qn,
        "fixation_count": fx_cnt,
        "saccade_count": sc_cnt,
        "fixation_duration_ms": fx_dur,
        "saccade_duration_ms": sc_dur,
        "fixation_amplitude_mean_deg": fx_amp_mean,
        "saccade_amplitude_mean_deg": sc_amp_mean,
        "saccade_maxvel_mean": sc_maxvel_mean,
        "saccade_meanvel_mean": sc_meanvel_mean,
        "kw_fix_duration_ms": kw_fx_dur,
        "inst_fix_duration_ms": inst_fx_dur,
        "bg_fix_duration_ms": bg_fx_dur,
        "kw_time_ratio": kw_ratio,
        "inst_time_ratio": inst_ratio,
        "bg_time_ratio": bg_ratio,
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
