"""
数据描述性统计分析工具
用于分析不同组别的眼动特征数据及标签分布
"""
import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def normalize_subject_id(subject_raw, group_type):
    """将 MMSE 表格中的 subject ID 转换为特征表中的标准格式"""
    match = re.search(r'\d+', str(subject_raw))
    if not match:
        return None
    num = int(match.group())

    if group_type == 'ad':
        return f"ad_group_{num}"
    elif group_type == 'control':
        return f"control_group_{num}"
    elif group_type == 'mci':
        return f"mci_group_{num}"
    return None


def load_mmse_scores():
    """加载 MMSE 评分数据，计算 Q1-Q5 的任务分"""
    mmse_dir = os.path.join(project_root(), "data", "MMSE_Score")

    files_map = {
        "ad_group.csv": "ad",
        "control_group.csv": "control",
        "mci_group.csv": "mci"
    }

    all_scores = []

    for fname, gtype in files_map.items():
        fpath = os.path.join(mmse_dir, fname)
        if not os.path.exists(fpath):
            continue

        try:
            df = pd.read_csv(fpath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(fpath, encoding='gbk')

        df.columns = [c.strip() for c in df.columns]
        subject_col = df.columns[0]

        for _, row in df.iterrows():
            sub_id = row[subject_col]
            std_id = normalize_subject_id(sub_id, gtype)
            if not std_id:
                continue

            q1_score = row.get('年份', 0) + row.get('季节', 0) + row.get('月份', 0) + row.get('星期', 0)
            q2_score = row.get('省市区', 0) + row.get('街道', 0) + row.get('建筑', 0) + row.get('楼层', 0)
            q3_score = row.get('即刻记忆', 0)
            q4_score = (row.get('100-7', 0) + row.get('93-7', 0) + row.get('86-7', 0) +
                        row.get('79-7', 0) + row.get('72-7', 0))
            q5_score = row.get('词1', 0) + row.get('词2', 0) + row.get('词3', 0)

            all_scores.append({
                'subject': std_id,
                'group': gtype,
                'q1_score': q1_score,
                'q2_score': q2_score,
                'q3_score': q3_score,
                'q4_score': q4_score,
                'q5_score': q5_score,
                'total_score': row.get('总分', 0)
            })

    return pd.DataFrame(all_scores)


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def describe_label_distribution(q_num, groups=None):
    """
    分析指定问题的标签（MMSE 分数）分布

    参数:
        q_num: 问题编号 (1-5)
        groups: 组别列表，默认 ["control", "mci", "ad"]
    """
    if groups is None:
        groups = ["control", "mci", "ad"]

    mmse_df = load_mmse_scores()
    target_col = f"q{q_num}_score"

    print(f"\n{'='*60}")
    print(f"Q{q_num} 标签分布 (MMSE 分数)")
    print(f"{'='*60}")

    # 整体分布
    print(f"\n【整体分布】 样本数: {len(mmse_df)}")
    scores = mmse_df[target_col]
    print(f"  均值: {scores.mean():.2f}")
    print(f"  标准差: {scores.std():.2f}")
    print(f"  范围: [{scores.min():.0f}, {scores.max():.0f}]")
    print(f"  唯一值数: {scores.nunique()}")

    # 值分布
    print(f"\n  值分布:")
    value_counts = scores.value_counts().sort_index()
    for val, cnt in value_counts.items():
        pct = cnt / len(scores) * 100
        bar = '█' * int(pct / 2)
        print(f"    {val:.0f}: {cnt:3d} ({pct:5.1f}%) {bar}")

    # 各组分布
    print(f"\n{'-'*40}")
    print("各组别标签分布:")
    print(f"{'-'*40}")

    group_stats = []
    for group in groups:
        group_df = mmse_df[mmse_df['group'] == group]
        if len(group_df) == 0:
            continue

        scores = group_df[target_col]
        stats = {
            'group': group.upper(),
            'n': len(group_df),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        group_stats.append(stats)

        print(f"\n【{group.upper()} 组】 n={len(group_df)}")
        print(f"  均值: {scores.mean():.2f}, 标准差: {scores.std():.2f}, 范围: [{scores.min():.0f}, {scores.max():.0f}]")

        value_counts = scores.value_counts().sort_index()
        for val, cnt in value_counts.items():
            pct = cnt / len(scores) * 100
            bar = '█' * int(pct / 2)
            print(f"    {val:.0f}: {cnt:2d} ({pct:5.1f}%) {bar}")

    # 组间对比表格
    if group_stats:
        print(f"\n{'-'*40}")
        print("组间对比汇总:")
        print(f"{'-'*40}")
        stats_df = pd.DataFrame(group_stats)
        stats_df = stats_df.set_index('group')
        print(stats_df.round(2))

    return mmse_df

def get_merged_features_path(group, q_num):
    """构建合并特征文件的路径"""
    return os.path.join(
        project_root(),
        "data", "MLP_data", "features", "merged_features",
        f"{group}_group",
        f"{group}_q{q_num}.csv"
    )

def describe_question_by_group(q_num, groups=None):
    """
    分析指定问题各组别的数据描述性统计

    参数:
        q_num: 问题编号 (1-5)
        groups: 组别列表，默认 ["control", "mci", "ad"]
    """
    if groups is None:
        groups = ["control", "mci", "ad"]

    print(f"\n{'='*60}")
    print(f"Q{q_num} 各组别数据描述性统计")
    print(f"{'='*60}")

    # 排除非特征列
    exclude_cols = ['group', 'subject', 'q', 'label', 'ROI_CAT']

    all_dfs = {}

    for group in groups:
        path = get_merged_features_path(group, q_num)
        if not os.path.exists(path):
            print(f"警告：未找到文件 {path}")
            continue

        df = pd.read_csv(path)
        all_dfs[group] = df

        # 获取特征列
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        print(f"\n{'-'*40}")
        print(f"【{group.upper()} 组】 样本数: {len(df)}")
        print(f"{'-'*40}")

        # 基本统计
        desc = df[feature_cols].describe().T
        desc['missing'] = df[feature_cols].isna().sum()
        desc['missing%'] = (df[feature_cols].isna().sum() / len(df) * 100).round(2)

        # 格式化输出
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: f'{x:.4f}')

        print(desc[['count', 'mean', 'std', 'min', '50%', 'max', 'missing', 'missing%']])

    # 组间对比
    if len(all_dfs) > 1:
        print(f"\n{'='*60}")
        print(f"Q{q_num} 组间均值对比")
        print(f"{'='*60}")

        comparison = pd.DataFrame()
        for group, df in all_dfs.items():
            feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
            comparison[group] = df[feature_cols].mean()

        print(comparison.round(4))

    return all_dfs

def describe_all_questions(include_labels=True):
    """分析所有问题的数据"""
    for q in range(1, 6):
        if include_labels:
            describe_label_distribution(q)
        describe_question_by_group(q)


if __name__ == "__main__":
    # 默认分析 Q2 的标签分布和特征统计
    describe_label_distribution(2)
    describe_question_by_group(2)
