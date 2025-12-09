import os
import pandas as pd
from typing import List, Dict

# 将事件特征与 RQA 特征按 (group, subject, q) 合并为统一的受试者级特征


def project_root():
    # 返回项目根目录：.../lsh_eye_analysis
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dir(p: str) -> str:
    # 若目录不存在则创建，返回该路径
    os.makedirs(p, exist_ok=True)
    return p


def read_csv_safe(fp: str) -> pd.DataFrame:
    # 读取 CSV，若不存在或读取失败则返回空表，保证后续合并流程稳健
    if not os.path.exists(fp):
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()


def merge_group_q(event_root: str, rqa_root: str, out_root: str, group: str, q: int, join: str) -> str:
    # 合并某组某题的事件特征与 RQA 特征，保存到 merged_features/<group>_group/<group>_q{q}.csv
    ev_fp = os.path.join(event_root, f"{group}_group", f"{group}_q{q}.csv")
    rq_fp = os.path.join(rqa_root, f"{group}_group", f"{group}_q{q}.csv")
    ev_df = read_csv_safe(ev_fp)
    rq_df = read_csv_safe(rq_fp)
    if ev_df.empty and rq_df.empty:
        out_fp = os.path.join(out_root, f"{group}_group", f"{group}_q{q}.csv")
        ensure_dir(os.path.dirname(out_fp))
        pd.DataFrame().to_csv(out_fp, index=False)
        return out_fp
    # 允许选择连接方式（inner/outer），默认使用 inner
    how = join if join in ("inner", "outer") else "inner"
    if ev_df.empty:
        merged = rq_df.copy()
    elif rq_df.empty:
        merged = ev_df.copy()
    else:
        # 按 group/subject/q 进行键连接
        merged = pd.merge(ev_df, rq_df, on=["group", "subject", "q"], how=how)
    out_fp = os.path.join(out_root, f"{group}_group", f"{group}_q{q}.csv")
    ensure_dir(os.path.dirname(out_fp))
    merged.to_csv(out_fp, index=False)
    return out_fp


def merge_all(groups: List[str], event_dir_name: str, rqa_dir_name: str, out_dir_name: str, join: str) -> Dict[str, Dict[int, str]]:
    # 批量合并多个组的 q1..q5 特征
    root = project_root()
    event_root = os.path.join(root, "data", "MLP_data", "features", event_dir_name)
    rqa_root = os.path.join(root, "data", "MLP_data", "features", rqa_dir_name)
    out_root = os.path.join(root, "data", "MLP_data", "features", out_dir_name)
    res: Dict[str, Dict[int, str]] = {}
    for g in groups:
        saved: Dict[int, str] = {}
        for q in [1, 2, 3, 4, 5]:
            saved[q] = merge_group_q(event_root, rqa_root, out_root, g, q, join)
        res[g] = saved
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="合并事件特征与 RQA 特征，生成统一受试者级特征表")
    parser.add_argument("--groups", type=str, default="control,mci,ad")
    parser.add_argument("--event-dir-name", type=str, default="event_features")
    parser.add_argument("--rqa-dir-name", type=str, default="rqa_features")
    parser.add_argument("--out-dir-name", type=str, default="merged_features")
    parser.add_argument("--join", type=str, default="inner", help="合并方式：inner 或 outer")
    args = parser.parse_args()
    groups = [s.strip() for s in args.groups.split(",") if s.strip()]
    result = merge_all(groups, args.event_dir_name, args.rqa_dir_name, args.out_dir_name, args.join)
    print(result)

