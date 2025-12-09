import os
import re
import sys
import pandas as pd
from typing import List, Dict

def project_root():
    """
    返回项目根目录（当前文件的上一级目录），用于构造统一的数据/模块路径。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_q_num_from_filename(filename: str) -> int:
    """
    从文件名中解析题号（Q1–Q5 的 q{digit} 部分），例如：
        xxx_q3_preprocessed_calibrated.csv -> 3
    若未匹配到，则默认返回 1。
    """
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1

def group_calibrated_root(calib_dir_name: str, group: str) -> str:
    """
    根据标定目录名和组别名称，拼出该组标定数据的根目录，例如：
        data/{calib_dir_name}/{group}_calibrated
    参数
    ----
    calib_dir_name : str
        标定结果所在的子目录名，如 data_calibration_mix 等。
    group : str
        组别名称，例如 "control" / "mci" / "ad"。
    """
    return os.path.join(project_root(), "data", calib_dir_name, f"{group}_calibrated")

def list_subject_folders_calibrated(root: str) -> List[str]:
    """
    列出标定根目录下所有受试者文件夹路径（只返回目录，不含文件）。
    root 形如：data/{calib_dir_name}/{group}_calibrated
    """
    if not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

def list_q_files_calibrated(folder: str) -> List[str]:
    """
    列出某个受试者目录中所有已标定的题目级文件：
        *_preprocessed_calibrated.csv
    返回这些文件的完整路径列表。
    """
    files = []
    for name in os.listdir(folder):
        if name.endswith("_preprocessed_calibrated.csv"):
            files.append(os.path.join(folder, name))
    return files

def ensure_dir(p: str) -> str:
    """
    确保目录 p 存在，若不存在则创建。返回该路径本身，便于链式调用。
    """
    os.makedirs(p, exist_ok=True)
    return p

def export_group_events(group: str, calib_dir_name: str) -> Dict[int, str]:
    """
    对指定组别 group，将所有受试者、所有题目（Q1–Q5）的事件级数据导出为按题目聚合的 CSV。

    数据来源：
        data/{calib_dir_name}/{group}_calibrated/{subject}/..._preprocessed_calibrated.csv

    处理流程：
        1. 遍历该组下所有受试者文件夹；
        2. 对每个题目文件调用 EventAnalyzer.process_single_file 得到事件级 DataFrame；
        3. 在 DataFrame 中添加 group / subject 字段；
        4. 对同组同题目下的所有受试者事件表做 concat；
        5. 写出到：
           data/MLP_data/event_data/{group}_group/{group}_q{q}.csv

    参数
    ----
    group : str
        组别名称，例如 "control" / "mci" / "ad"。
    calib_dir_name : str
        标定结果子目录名，例如 "data_calibration" / "data_calibration_mix" 等。

    返回
    ----
    Dict[int, str]
        key 为题号 q（1–5），value 为导出 CSV 文件的路径。
    """
    # 动态导入 EventAnalyzer，避免在模块顶部就依赖 analysis 包
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    analyzer = EventAnalyzer()

    # 为每个题目准备一个列表，用于收集该组所有受试者的事件表
    q_events: Dict[int, List[pd.DataFrame]] = {1: [], 2: [], 3: [], 4: [], 5: []}

    # 当前组别的标定数据根目录，例如 data/data_calibration_mix/control_calibrated
    root = group_calibrated_root(calib_dir_name, group)
    subjects = list_subject_folders_calibrated(root)

    for subj in subjects:
        # subj 形如：.../control_calibrated/subject_001
        for fp in list_q_files_calibrated(subj):
            # 解析题号 q（1–5）
            q = parse_q_num_from_filename(os.path.basename(fp))
            if q not in q_events:
                continue
            # 通过 EventAnalyzer 得到单题文件的事件级数据
            evt_df, _ = analyzer.process_single_file(fp, debug=False)
            if evt_df is None or evt_df.empty:
                continue
            # 标记 group / subject 信息，方便后续建模区分来源
            evt_df["group"] = group
            evt_df["subject"] = os.path.basename(subj)
            q_events[q].append(evt_df)

    # 输出目录：按组别单独存放一套事件数据
    # 例如：data/MLP_data/event_data/control_group/control_q1.csv
    out_dir = ensure_dir(
        os.path.join(project_root(), "data", "MLP_data", "event_data", f"{group}_group")
    )

    saved: Dict[int, str] = {}
    for q in [1, 2, 3, 4, 5]:
        if q_events[q]:
            # 将该组该题目下所有受试者的事件表拼在一起
            df_out = pd.concat(q_events[q], ignore_index=True)
        else:
            # 若没有任何数据，写出一个空表占位，便于后续脚本统一处理
            df_out = pd.DataFrame()
        out_fp = os.path.join(out_dir, f"{group}_q{q}.csv")
        df_out.to_csv(out_fp, index=False)
        saved[q] = out_fp

    return saved

def export_all(groups: List[str], calib_dir_name: str) -> Dict[str, Dict[int, str]]:
    """
    对多个组别批量导出事件数据。

    参数
    ----
    groups : List[str]
        需要导出的组别列表，例如 ["control", "mci", "ad"]。
    calib_dir_name : str
        标定目录名称，同 export_group_events 的 calib_dir_name。

    返回
    ----
    Dict[str, Dict[int, str]] :
        最外层 key 为组别名，内层字典为 {q: csv_path}。
    """
    res: Dict[str, Dict[int, str]] = {}
    for g in groups:
        res[g] = export_group_events(g, calib_dir_name)
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Export eye events per group/question to CSV for MLP"
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="control,mci,ad",
        help="逗号分隔的组别名称列表，例如 control,mci,ad",
    )
    parser.add_argument(
        "--calibrated-dir-name",
        type=str,
        default="data_calibration_mix",
        help="标定结果所在目录名，如: data_calibration | data_calibration_mix | data_calibration_gradient_ascent",
    )
    args = parser.parse_args()

    # 解析 groups 参数，去除空白项
    groups = [s.strip() for s in args.groups.split(",") if s.strip()]

    # 执行按组别导出
    result = export_all(groups, calib_dir_name=args.calibrated_dir_name)
    print(result)
