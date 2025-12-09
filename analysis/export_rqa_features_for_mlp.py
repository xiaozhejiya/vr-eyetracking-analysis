import os
import re
import sys
import pandas as pd
import numpy as np
from typing import List, Dict

# 将项目根（lsh_eye_analysis）加入搜索路径，确保可导入 config 和 analysis 模块

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import Config

def project_root():
    # 返回项目根目录：.../lsh_eye_analysis
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(p: str) -> str:
    # 若目录不存在则创建，返回该路径
    os.makedirs(p, exist_ok=True)
    return p


def parse_q_num_from_filename(filename: str) -> int:
    # 依据文件名中的 "q{数字}" 提取题号，例如 ad9q5_preprocessed_calibrated.csv -> 5
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1


def list_subject_folders_calibrated(root: str) -> List[str]:
    # 列出某组校准根目录下的受试者子目录
    if not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def list_q_files(folder: str) -> List[str]:
    # 列出受试者目录中所有校准后的题目 CSV 文件
    files = []
    if not os.path.exists(folder):
        return files
    for name in os.listdir(folder):
        if name.endswith("_preprocessed_calibrated.csv"):
            files.append(os.path.join(folder, name))
    return files


def group_calibrated_root(calib_dir_name: str, group: str) -> str:
    # 某组的校准数据根，例如 data/data_calibration_mix/ad_calibrated
    return os.path.join(project_root(), "data", calib_dir_name, f"{group}_calibrated")


def export_group_rqa_features(group: str, calib_dir_name: str, params: Dict) -> Dict[int, str]:
    # 遍历该组的所有受试者与题号，计算 RQA 指标，并按 q1..q5 分别输出受试者级特征表
    sys.path.append(project_root())
    from analysis.rqa_analyzer import RQAAnalyzer

    rqa = RQAAnalyzer()
    # 每个题号收集一组受试者的指标行
    q_rows: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    root = group_calibrated_root(calib_dir_name, group)
    subjects = list_subject_folders_calibrated(root)

    for subj in subjects:
        subj_name = os.path.basename(subj)
        for fp in list_q_files(subj):
            q = parse_q_num_from_filename(os.path.basename(fp))
            if q not in q_rows:
                continue
            # 计算单文件的 RQA 指标（1D-x 与 2D-xy）
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
            # 添加该受试者该题号的一行记录
            q_rows[q].append(row)

    # 输出目录：data/MLP_data/features/rqa_features/<group>_group
    out_dir = ensure_dir(os.path.join(project_root(), "data", "MLP_data", "features", "rqa_features", f"{group}_group"))
    saved: Dict[int, str] = {}
    for q in [1, 2, 3, 4, 5]:
        rows = q_rows[q]
        if rows:
            df = pd.DataFrame(rows)
            # 若同一受试者同一题号有多条记录（多文件），按 subject 聚合求均值
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
        # 保存该题号的受试者级特征表
        df.to_csv(out_fp, index=False)
        saved[q] = out_fp
    return saved


def export_all(groups: List[str], calib_dir_name: str, params: Dict) -> Dict[str, Dict[int, str]]:
    # 批量导出多个组的 RQA 特征
    res: Dict[str, Dict[int, str]] = {}
    for g in groups:
        res[g] = export_group_rqa_features(g, calib_dir_name, params)
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="按组与题号导出 RQA 受试者级特征到 CSV")
    parser.add_argument("--groups", type=str, default="control,mci,ad")
    parser.add_argument("--calibrated-dir-name", type=str, default="data_calibration_mix",
                        help="选择校准数据目录：data_calibration | data_calibration_mix | data_calibration_gradient_ascent")
    parser.add_argument("--m", type=int, default=Config.RQA_DEFAULT_PARAMS['m'])
    parser.add_argument("--tau", type=int, default=Config.RQA_DEFAULT_PARAMS['tau'])
    parser.add_argument("--eps", type=float, default=Config.RQA_DEFAULT_PARAMS['eps'])
    parser.add_argument("--lmin", type=int, default=Config.RQA_DEFAULT_PARAMS['lmin'])
    args = parser.parse_args()

    groups = [s.strip() for s in args.groups.split(",") if s.strip()]
    # 将命令行参数裁剪到配置允许的范围内
    def clamp(name, value):
        r = Config.RQA_PARAM_RANGES[name]
        return max(r['min'], min(r['max'], value))
    params = {
        "m": clamp('m', args.m),
        "tau": clamp('tau', args.tau),
        "eps": clamp('eps', args.eps),
        "lmin": clamp('lmin', args.lmin),
    }
    # 执行导出
    result = export_all(groups, args.calibrated_dir_name, params)
    print(result)
