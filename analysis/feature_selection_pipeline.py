import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 将项目根目录添加到路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import Config

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_merged_features_path(group, q_num):
    """
    构建合并特征文件的路径。
    例如：data/MLP_data/features/merged_features/ad_group/ad_q1.csv
    """
    return os.path.join(
        project_root(), 
        "data", "MLP_data", "features", "merged_features", 
        f"{group}_group", 
        f"{group}_q{q_num}.csv"
    )

def normalize_subject_id(subject_raw, group_type):
    """
    将 MMSE 表格中的 subject ID (如 'ad01', 'n01', 'M01') 
    转换为特征表中的标准格式 (如 'ad_group_1', 'control_group_1', 'mci_group_1')。
    """
    # 提取数字部分
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
    """
    加载 data/MMSE_Score 下的所有组别分数，并按规则计算 Q1-Q5 的任务分。
    返回一个大 DataFrame，索引为 standardized_subject_id，列包含 q1_score, q2_score...
    """
    mmse_dir = os.path.join(project_root(), "data", "MMSE_Score")
    
    # 文件名映射到 group_type
    files_map = {
        "ad_group.csv": "ad",
        "control_group.csv": "control",
        "mci_group.csv": "mci"
    }
    
    all_scores = []
    
    for fname, gtype in files_map.items():
        fpath = os.path.join(mmse_dir, fname)
        if not os.path.exists(fpath):
            print(f"警告：未找到 MMSE 文件 {fpath}")
            continue
            
        # 读取 CSV，注意列名可能有微小差异，统一去除空格
        try:
            # 尝试 utf-8，失败则 gbk
            df = pd.read_csv(fpath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(fpath, encoding='gbk')
            
        # 统一列名：去除空白
        df.columns = [c.strip() for c in df.columns]
        
        # 找到“受试者”列（第一列）
        subject_col = df.columns[0]
        
        for _, row in df.iterrows():
            sub_id = row[subject_col]
            std_id = normalize_subject_id(sub_id, gtype)
            if not std_id:
                continue
                
            # 计算各题分数
            # Q1 时间定向: 年份+季节+月份+星期 (0-5)
            q1_score = row.get('年份', 0) + row.get('季节', 0) + row.get('月份', 0) + row.get('星期', 0)
            
            # Q2 空间定向: 省市区+街道+建筑+楼层 (0-5)
            # 注意：省市区 可能是 0-2 分
            q2_score = row.get('省市区', 0) + row.get('街道', 0) + row.get('建筑', 0) + row.get('楼层', 0)
            
            # Q3 即刻记忆: (0-3)
            q3_score = row.get('即刻记忆', 0)
            
            # Q4 注意与计算: 5个步骤 (0-5)
            q4_score = (row.get('100-7', 0) + row.get('93-7', 0) + row.get('86-7', 0) + 
                        row.get('79-7', 0) + row.get('72-7', 0))
            
            # Q5 延迟回忆: 3个词 (0-3)
            q5_score = row.get('词1', 0) + row.get('词2', 0) + row.get('词3', 0)
            
            all_scores.append({
                'subject': std_id,
                'q1_score': q1_score,
                'q2_score': q2_score,
                'q3_score': q3_score,
                'q4_score': q4_score,
                'q5_score': q5_score,
                'total_score': row.get('总分', 0)
            })
            
    return pd.DataFrame(all_scores)

def load_data_for_question_regression(q_num, mmse_df):
    """
    加载特征并与 MMSE 分数合并。
    """
    groups = ["control", "mci", "ad"]
    dfs = []
    
    target_col = f"q{q_num}_score"
    if target_col not in mmse_df.columns:
        print(f"错误：MMSE 数据中没有 {target_col}")
        return None
        
    for group in groups:
        path = get_merged_features_path(group, q_num)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 合并分数
            # 左连接：保留特征表中的所有行，匹配 MMSE 分数
            merged = pd.merge(df, mmse_df[['subject', target_col]], on='subject', how='inner')
            
            if len(merged) < len(df):
                print(f"警告：组 {group} Q{q_num} 有 {len(df)-len(merged)} 个受试者未匹配到 MMSE 分数")
                
            dfs.append(merged)
        else:
            print(f"警告：未找到文件：{path}")
            
    if not dfs:
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df, target_col

def filter_high_correlation(X, threshold=0.95):
    """
    移除高度相关的特征。
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_selected = X.drop(columns=to_drop)
    return X_selected, to_drop

def select_features_elasticnet(X, y, cv=None, min_features=2):
    """
    使用 ElasticNet 回归进行特征选择。
    ElasticNet 结合了 L1 和 L2 正则化，比 Lasso 在处理共线性特征时更稳定。

    改进：
    - 显式指定 alpha 搜索范围，避免自动范围过大导致所有系数为 0
    - 当选出特征过少时，自动回退到更小的 alpha 或使用相关性排序
    - 支持传入分层 CV (StratifiedKFold)，确保各 fold 中组别比例一致

    参数：
    - X: 特征矩阵
    - y: 目标变量
    - cv: 交叉验证策略，可以是 int 或 CV splitter 对象 (如 StratifiedKFold 的 splits)
    - min_features: 最少选择的特征数量，不足时用相关性补充
    """
    # l1_ratio 控制 L1 和 L2 的比例
    l1_ratios = [.1, .5, .7, .9, .95, .99, 1]

    # 显式指定 alpha 搜索范围（从小到大），避免 sklearn 自动范围过大
    # 范围从 1e-4 到 10，共 100 个点（对数均匀分布）
    alphas = np.logspace(-4, 1, 100)

    # 默认使用 5 折 CV
    if cv is None:
        cv = 5

    # ElasticNetCV 自动选择最佳 alpha 和 l1_ratio
    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv,
        random_state=42,
        max_iter=20000
    )
    enet.fit(X, y)

    # 获取系数绝对值
    coefs = np.abs(enet.coef_)

    # 找出系数非 0 的特征
    non_zero_indices = np.where(coefs > 1e-5)[0]

    # 按系数大小排序
    sorted_indices = non_zero_indices[np.argsort(coefs[non_zero_indices])[::-1]]

    selected_features = [X.columns[i] for i in sorted_indices]
    importances = coefs[sorted_indices]

    # 回退机制：如果选出的特征太少，使用与目标的相关性排序来补充
    if len(selected_features) < min_features:
        print(f"  [回退] ElasticNet 仅选出 {len(selected_features)} 个特征，使用相关性排序补充至 {min_features} 个")
        # 计算每个特征与目标的绝对相关性
        correlations = X.corrwith(y).abs().fillna(0)
        # 排除已选特征
        remaining_features = [f for f in X.columns if f not in selected_features]
        # 按相关性排序
        remaining_sorted = correlations[remaining_features].sort_values(ascending=False)
        # 补充到 min_features 个
        n_to_add = min_features - len(selected_features)
        backup_features = remaining_sorted.head(n_to_add).index.tolist()
        selected_features = selected_features + backup_features
        # 更新 importances (相关性值)
        backup_importances = correlations[backup_features].values
        importances = np.concatenate([importances, backup_importances]) if len(importances) > 0 else backup_importances

    return selected_features, importances, enet.alpha_, enet.l1_ratio_


def stability_selection_classification(X, y_binary, n_bootstrap=50, sample_ratio=0.7,
                                        threshold=0.6, min_features=3, max_features=6):
    """
    使用稳定性选择 (Stability Selection) 进行分类特征筛选。

    原理：在多个 bootstrap 样本上运行 L1 正则化的 Logistic 回归，
    统计每个特征被选中的频率，选择频率高于阈值的特征。

    参数:
        X: 特征 DataFrame (已标准化)
        y_binary: 二分类标签 (0/1)
        n_bootstrap: bootstrap 采样次数
        sample_ratio: 每次采样的样本比例
        threshold: 特征被选中的频率阈值 (默认 0.6，即 60% 的采样中被选中)
        min_features: 最少选择的特征数
        max_features: 最多选择的特征数

    返回:
        selected_features: 稳定特征列表
        selection_freq: 各特征的选择频率字典
    """
    n_samples, n_features = X.shape
    feature_names = X.columns.tolist()

    # 记录每个特征被选中的次数
    selection_counts = np.zeros(n_features)

    # 使用多个 alpha 值增加多样性
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]

    for i in range(n_bootstrap):
        # Bootstrap 采样
        np.random.seed(i)
        sample_size = int(n_samples * sample_ratio)
        sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)

        X_sample = X.iloc[sample_idx].values
        y_sample = y_binary[sample_idx]

        # 检查是否两个类都有样本
        if len(np.unique(y_sample)) < 2:
            continue

        # 随机选择一个 alpha
        alpha = alphas[i % len(alphas)]

        try:
            # L1 正则化的 Logistic 回归
            clf = LogisticRegression(
                penalty='l1',
                solver='saga',
                C=1.0/alpha,  # sklearn 中 C = 1/alpha
                max_iter=5000,
                random_state=i,
                class_weight='balanced'
            )
            clf.fit(X_sample, y_sample)

            # 记录非零系数的特征
            non_zero_idx = np.where(np.abs(clf.coef_[0]) > 1e-5)[0]
            selection_counts[non_zero_idx] += 1
        except Exception as e:
            continue

    # 计算选择频率
    selection_freq = selection_counts / n_bootstrap
    freq_dict = {feature_names[i]: selection_freq[i] for i in range(n_features)}

    # 按频率排序
    sorted_features = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 选择频率高于阈值的特征
    stable_features = [f for f, freq in sorted_features if freq >= threshold]

    # 如果稳定特征太少，降低阈值或取 top-K
    if len(stable_features) < min_features:
        stable_features = [f for f, _ in sorted_features[:min_features]]
        print(f"  [稳定性选择] 阈值 {threshold} 下仅有 {len([f for f, freq in sorted_features if freq >= threshold])} 个特征，"
              f"自动选取 Top {min_features}")

    # 如果稳定特征太多，截取
    if len(stable_features) > max_features:
        stable_features = stable_features[:max_features]

    print(f"  [稳定性选择] 结果 (n_bootstrap={n_bootstrap}, threshold={threshold}):")
    for f in stable_features:
        print(f"    - {f}: {freq_dict[f]*100:.1f}%")

    return stable_features, freq_dict


def check_target_distribution(y, target_name):
    """
    检查目标变量的分布，并给出警告。
    返回：(is_problematic, info_dict)
    """
    y_vals = y.values if hasattr(y, 'values') else y
    unique_vals = np.unique(y_vals)
    value_counts = pd.Series(y_vals).value_counts()

    info = {
        "mean": float(np.mean(y_vals)),
        "std": float(np.std(y_vals)),
        "min": float(np.min(y_vals)),
        "max": float(np.max(y_vals)),
        "n_unique": len(unique_vals),
        "most_common_value": value_counts.index[0],
        "most_common_count": int(value_counts.iloc[0]),
        "most_common_pct": float(value_counts.iloc[0] / len(y_vals) * 100)
    }

    is_problematic = False
    warnings = []

    # 检查 1：方差是否过小（相对于均值）
    if info["std"] < 0.1:
        warnings.append(f"方差过小 (std={info['std']:.4f})，回归可能不稳定")
        is_problematic = True

    # 检查 2：是否有太多相同值
    if info["most_common_pct"] > 70:
        warnings.append(f"目标分布高度不均：{info['most_common_pct']:.1f}% 的样本值为 {info['most_common_value']}")
        is_problematic = True

    # 检查 3：唯一值是否太少（对于回归来说）
    if info["n_unique"] <= 3:
        warnings.append(f"目标变量仅有 {info['n_unique']} 个唯一值，可能更适合分类任务")
        is_problematic = True

    info["warnings"] = warnings
    return is_problematic, info


def evaluate_subset_regressor(X, y, feature_names, model_type='rf', cv=None, score_range=None):
    """
    使用交叉验证评估回归效果 (R^2 Score 和 MAE)

    参数:
        score_range: (min, max) 元组，用于将预测值裁剪并四舍五入到合法分值范围
    """
    if not feature_names:
        print("  [警告] 特征列表为空，跳过评估")
        return np.nan, np.nan, np.nan

    X_subset = X[feature_names].values
    y_vals = y.values if hasattr(y, 'values') else y

    # 交叉验证策略
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 构建 Pipeline 以防止数据泄露
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]

    if model_type == 'rf':
        steps = [('imputer', SimpleImputer(strategy='mean'))]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lasso':
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
    elif model_type == 'ridge':
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(steps + [('model', model)])

    # 手动交叉验证以支持离散化后的 MAE
    r2_scores = []
    mae_raw_scores = []
    mae_discrete_scores = []

    cv_iter = cv if isinstance(cv, list) else list(cv.split(X_subset, y_vals))

    for train_idx, test_idx in cv_iter:
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y_vals[train_idx], y_vals[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # R² 计算
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_scores.append(r2)

        # 原始 MAE
        mae_raw = np.mean(np.abs(y_test - y_pred))
        mae_raw_scores.append(mae_raw)

        # 离散化后的 MAE
        if score_range is not None:
            y_pred_discrete = np.clip(y_pred, score_range[0], score_range[1])
            y_pred_discrete = np.round(y_pred_discrete)
            mae_discrete = np.mean(np.abs(y_test - y_pred_discrete))
        else:
            mae_discrete = mae_raw
        mae_discrete_scores.append(mae_discrete)

    return np.mean(r2_scores), np.mean(mae_raw_scores), np.mean(mae_discrete_scores)


def evaluate_two_stage_model(X, y, feature_names, cv=None, max_score=5, cls_threshold=0.5,
                              reg_feature_names=None):
    """
    两阶段模型评估（分类+回归）:
    - Head1: 二分类预测是否满分
    - Head2: 在非满分样本上回归缺失分 d = max_score - y

    参数:
        X: 特征 DataFrame
        y: 目标变量 (原始分数)
        feature_names: 分类阶段使用的特征列表 (共享特征)
        cv: 交叉验证策略
        max_score: 满分值 (默认 5)
        cls_threshold: 分类阈值 (默认 0.5)
        reg_feature_names: 回归阶段使用的特征列表，默认与分类相同

    返回:
        metrics: 包含各项评估指标的字典
    """
    if not feature_names:
        print("  [警告] 特征列表为空，跳过评估")
        return None

    # 回归阶段特征默认与分类相同
    if reg_feature_names is None:
        reg_feature_names = feature_names

    X_cls = X[feature_names].values
    X_reg = X[reg_feature_names].values
    y_vals = y.values if hasattr(y, 'values') else np.array(y)

    # 构建二分类标签: 是否满分
    y_is_max = (y_vals == max_score).astype(int)

    # 构建回归目标: 缺失分 d = max_score - y (仅对非满分有意义)
    y_deficit = max_score - y_vals

    cv_iter = cv if isinstance(cv, list) else list(cv.split(X_cls, y_vals))

    # 记录各 fold 的指标
    cls_accuracies = []
    cls_f1s = []
    cls_aucs = []
    final_maes = []
    final_mae_discretes = []

    print(f"\n  [两阶段模型] 满分样本: {np.sum(y_is_max)}/{len(y_vals)} ({100*np.mean(y_is_max):.1f}%)")
    if list(feature_names) != list(reg_feature_names):
        print(f"  [两阶段模型] 分类特征: {feature_names}")
        print(f"  [两阶段模型] 回归特征: {reg_feature_names}")

    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        # 分类阶段用 X_cls
        X_cls_train, X_cls_test = X_cls[train_idx], X_cls[test_idx]
        # 回归阶段用 X_reg
        X_reg_train, X_reg_test = X_reg[train_idx], X_reg[test_idx]

        y_train_is_max, y_test_is_max = y_is_max[train_idx], y_is_max[test_idx]
        y_train_deficit, y_test_deficit = y_deficit[train_idx], y_deficit[test_idx]
        y_test_true = y_vals[test_idx]

        # ========== Head1: 分类器 (是否满分) ==========
        cls_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegressionCV(cv=3, random_state=42, max_iter=2000, class_weight='balanced'))
        ])
        cls_pipeline.fit(X_cls_train, y_train_is_max)
        y_pred_proba = cls_pipeline.predict_proba(X_cls_test)[:, 1]
        y_pred_is_max = (y_pred_proba >= cls_threshold).astype(int)

        # 分类指标
        cls_acc = accuracy_score(y_test_is_max, y_pred_is_max)
        cls_f1 = f1_score(y_test_is_max, y_pred_is_max, zero_division=0)
        try:
            cls_auc = roc_auc_score(y_test_is_max, y_pred_proba)
        except ValueError:
            cls_auc = 0.5  # 只有一个类时

        cls_accuracies.append(cls_acc)
        cls_f1s.append(cls_f1)
        cls_aucs.append(cls_auc)

        # ========== Head2: 回归器 (缺失分) ==========
        # 仅在训练集的非满分样本上训练
        non_max_train_mask = y_train_is_max == 0
        if np.sum(non_max_train_mask) < 3:
            # 非满分样本太少，直接用均值预测
            reg_pred_deficit = np.full(len(X_reg_test), np.mean(y_train_deficit[non_max_train_mask]) if np.sum(non_max_train_mask) > 0 else 1)
        else:
            reg_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('reg', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]))
            ])
            reg_pipeline.fit(X_reg_train[non_max_train_mask], y_train_deficit[non_max_train_mask])
            reg_pred_deficit = reg_pipeline.predict(X_reg_test)

        # ========== 两阶段融合推理 ==========
        y_pred_final = np.zeros(len(X_cls_test))
        for i in range(len(X_cls_test)):
            if y_pred_proba[i] >= cls_threshold:
                # 预测为满分
                y_pred_final[i] = max_score
            else:
                # 预测为非满分，用回归结果
                # 缺失分 d 预测后转换为分数: score = max_score - d
                pred_deficit = np.clip(reg_pred_deficit[i], 1, max_score - 2)  # d in [1, 3] -> score in [2, 4]
                y_pred_final[i] = max_score - pred_deficit

        # 离散化
        y_pred_discrete = np.clip(np.round(y_pred_final), max_score - 3, max_score)  # [2, 5]

        # 计算 MAE
        mae = np.mean(np.abs(y_test_true - y_pred_final))
        mae_discrete = np.mean(np.abs(y_test_true - y_pred_discrete))

        final_maes.append(mae)
        final_mae_discretes.append(mae_discrete)

    metrics = {
        "cls_accuracy": float(np.mean(cls_accuracies)),
        "cls_f1": float(np.mean(cls_f1s)),
        "cls_auc": float(np.mean(cls_aucs)),
        "mae_raw": float(np.mean(final_maes)),
        "mae_discrete": float(np.mean(final_mae_discretes)),
        "threshold": cls_threshold,
        "max_score": max_score
    }

    print(f"  [Head1 分类] Acc: {metrics['cls_accuracy']:.4f}, F1: {metrics['cls_f1']:.4f}, AUC: {metrics['cls_auc']:.4f}")
    print(f"  [两阶段融合] MAE (原始): {metrics['mae_raw']:.4f}, MAE (离散化): {metrics['mae_discrete']:.4f}")

    return metrics


def main():
    output_dir = os.path.join(project_root(), "data", "MLP_data", "selected_features_regression")
    os.makedirs(output_dir, exist_ok=True)

    # 各问题的合法分值范围 (min, max)
    SCORE_RANGES = {
        1: (0, 5),   # Q1 时间定向: 年份+季节+月份+星期
        2: (0, 5),   # Q2 空间定向: 省市区+街道+建筑+楼层
        3: (0, 3),   # Q3 即刻记忆
        4: (0, 5),   # Q4 注意与计算
        5: (0, 3),   # Q5 延迟回忆
    }

    # 1. 预先加载并计算好所有受试者的 MMSE 分数
    print("正在加载并解析 MMSE 评分数据...")
    mmse_df = load_mmse_scores()
    print(f"共加载了 {len(mmse_df)} 名受试者的 MMSE 分数。")
    exclude_cols = ['group', 'subject', 'q', 'label', 'ROI_CAT',
                    'q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score', 'total_score']

    for q in range(1, 6):
        print(f"\n{'='*50}")
        print(f"正在处理问题 Q{q} (回归目标: MMSE Q{q} 得分)...")
        
        # 2. 加载特征并合并分数
        res = load_data_for_question_regression(q, mmse_df)
        if not res:
            print(f"Q{q} 数据加载失败，跳过。")
            continue
        df, target_col = res
        
        if len(df) < 10:
            print(f"由于有效样本不足 ({len(df)})，跳过 Q{q}。")
            continue
            
        # 准备 X 和 y
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols]
        y = df[target_col]
        
        # --- 2.1 ROI 特征裁剪 (Compositional Data Fix) ---
        # 移除冗余的绝对时长和部分比例特征，解决成分数据共线性问题
        roi_drop_cols = ["kw_fix_duration_ms", "inst_fix_duration_ms", "bg_fix_duration_ms", "bg_time_ratio"]
        existing_drop_cols = [c for c in roi_drop_cols if c in X.columns]
        if existing_drop_cols:
            print(f"执行 ROI 特征裁剪，移除: {existing_drop_cols}")
            X = X.drop(columns=existing_drop_cols)
        
        print(f"目标变量：{target_col}")
        print(f"样本数量：{len(X)}")
        print(f"初始特征数量：{X.shape[1]}")

        # --- 目标变量分布诊断 ---
        is_problematic, target_info = check_target_distribution(y, target_col)
        print(f"目标分布：mean={target_info['mean']:.2f}, std={target_info['std']:.2f}, "
              f"unique={target_info['n_unique']}, range=[{target_info['min']:.0f}, {target_info['max']:.0f}]")
        if is_problematic:
            for warn in target_info['warnings']:
                print(f"  [!] {warn}")
        
        # --- 2.2 构建分层 CV (Stratified CV) ---
        # 从 subject ID 中提取 group (例如 'ad_group_1' -> 'ad')
        # 用于 StratifiedKFold 保证各 fold 中组别比例一致
        group_labels = df['subject'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'unknown')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(skf.split(X, group_labels))
        
        # --- 2.3 数据预处理 (用于特征筛选步骤) ---
        # 注意：这里的 imputed/scaled 数据仅用于计算相关性和 RF/Lasso 筛选特征
        # 最终评估时会使用 Pipeline 在 CV 内部重新处理，避免泄露

        # 移除全 NaN 的列（无法用均值填充）
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"移除全 NaN 的特征: {all_nan_cols}")
            X = X.drop(columns=all_nan_cols)

        imputer_global = SimpleImputer(strategy='mean')
        X_imputed_val = imputer_global.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed_val, columns=X.columns)
        
        scaler_global = StandardScaler()
        X_scaled_val = scaler_global.fit_transform(X_imputed_df)
        X_scaled_df = pd.DataFrame(X_scaled_val, columns=X.columns)
        
        # 3. 相关性过滤
        print("步骤 1：过滤高相关性特征 (>0.95)...")
        # 使用全局标准化后的数据计算相关性
        X_uncorr_df, dropped_corr = filter_high_correlation(X_scaled_df, threshold=0.95)
        print(f"因共线性移除了 {len(dropped_corr)} 个特征。")
        print(f"剩余特征数量：{X_uncorr_df.shape[1]}")
        
        # 4. 特征筛选 (使用 ElasticNet)
        N_FEATURES = 5
        
        print(f"\n[筛选方法] ElasticNet (L1+L2 正则) + StratifiedKFold...")
        enet_feats, enet_weights, best_alpha, best_l1_ratio = select_features_elasticnet(
            X_uncorr_df, y, cv=cv_splits, min_features=2
        )
        print(f"  ElasticNet 选择了 {len(enet_feats)} 个非零特征 (Alpha: {best_alpha:.4f}, L1 Ratio: {best_l1_ratio:.2f})")
        
        enet_feats_top = enet_feats[:N_FEATURES]
        print(f"  ElasticNet 选出的 Top {len(enet_feats_top)} 特征: {enet_feats_top[:5]}...")

        # 获取当前问题的分值范围
        score_range = SCORE_RANGES[q]

        # 统一使用 RidgeCV + Pipeline + StratifiedCV 评估
        r2, mae_raw, mae_discrete = evaluate_subset_regressor(
            X, y, enet_feats_top, model_type='ridge', cv=cv_splits, score_range=score_range
        )

        # 计算基线 MAE
        y_vals = y.values if hasattr(y, 'values') else y
        y_mean = np.mean(y_vals)
        y_median = np.median(y_vals)

        # Mean baseline MAE (用于 R² 对标)
        baseline_mae_mean = np.mean(np.abs(y_vals - y_mean))

        # Median baseline MAE (更公平的 MAE 基线)
        baseline_mae_median = np.mean(np.abs(y_vals - y_median))

        # 离散化后的 baseline MAE
        y_mean_discrete = np.clip(np.round(y_mean), score_range[0], score_range[1])
        y_median_discrete = np.clip(np.round(y_median), score_range[0], score_range[1])
        baseline_mae_mean_discrete = np.mean(np.abs(y_vals - y_mean_discrete))
        baseline_mae_median_discrete = np.mean(np.abs(y_vals - y_median_discrete))

        r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        mae_raw_display = f"{mae_raw:.4f}" if not np.isnan(mae_raw) else "N/A"
        mae_discrete_display = f"{mae_discrete:.4f}" if not np.isnan(mae_discrete) else "N/A"

        print(f"  [统一评估] ElasticNet 特征集 (RidgeCV):")
        print(f"      R^2: {r2_display}")
        print(f"      MAE (原始): {mae_raw_display}")
        print(f"      MAE (离散化到 {score_range}): {mae_discrete_display}")
        print(f"  [基线参考]")
        print(f"      Mean baseline  -> MAE: {baseline_mae_mean:.4f}, 离散化: {baseline_mae_mean_discrete:.4f}")
        print(f"      Median baseline -> MAE: {baseline_mae_median:.4f}, 离散化: {baseline_mae_median_discrete:.4f}")

        if not np.isnan(r2) and r2 < 0:
            print(f"  [!] R^2 为负数，模型表现不如预测均值，可能因为：")
            print(f"      - 目标变量方差过小或分布高度不均")
            print(f"      - 特征与目标之间缺乏线性关系")
            print(f"      - 样本量过小导致交叉验证不稳定")

        # ========== Q2 特殊处理: 两阶段模型 (分类+回归) + 稳定性选择 ==========
        two_stage_metrics = None
        stability_info = None
        if q == 2:
            print(f"\n  [Q2 特殊处理] 使用稳定性选择筛选两阶段模型特征")

            # 构建二分类标签: 是否满分 (用于稳定性选择)
            y_is_max = (y_vals == 5).astype(int)

            # 稳定性选择：在分类任务上筛选稳定特征
            stable_feats, freq_dict = stability_selection_classification(
                X_scaled_df[X_uncorr_df.columns],  # 使用去相关后的特征
                y_is_max,
                n_bootstrap=50,
                sample_ratio=0.7,
                threshold=0.5,  # 50% 的采样中被选中
                min_features=3,
                max_features=6
            )

            # 记录稳定性选择信息
            stability_info = {
                "method": "stability_selection_classification",
                "n_bootstrap": 50,
                "threshold": 0.5,
                "stable_features": stable_feats,
                "selection_frequencies": {f: round(freq_dict[f], 3) for f in stable_feats}
            }

            # 两阶段模型评估：分类和回归使用相同的稳定特征
            print(f"\n  [Q2 两阶段模型] 使用稳定性选择特征 ({len(stable_feats)} 个)")
            two_stage_metrics = evaluate_two_stage_model(
                X, y, stable_feats, cv=cv_splits, max_score=5, cls_threshold=0.5,
                reg_feature_names=stable_feats  # 回归阶段使用相同特征
            )

            # Q2 最终使用稳定性选择的特征
            final_feats = stable_feats
        else:
            final_feats = enet_feats_top
        # 处理 NaN 值 (JSON 不支持 NaN)
        r2_safe = None if np.isnan(r2) else float(r2)
        mae_raw_safe = None if np.isnan(mae_raw) else float(mae_raw)
        mae_discrete_safe = None if np.isnan(mae_discrete) else float(mae_discrete)
        final_metrics = {
            "r2": r2_safe,
            "mae_raw": mae_raw_safe,
            "mae_discrete": mae_discrete_safe,
            "baseline_mae_mean": float(baseline_mae_mean),
            "baseline_mae_median": float(baseline_mae_median),
            "baseline_mae_mean_discrete": float(baseline_mae_mean_discrete),
            "baseline_mae_median_discrete": float(baseline_mae_median_discrete),
            "score_range": list(score_range),
            "method": "stability_selection" if q == 2 else "elasticnet"
        }

        # 如果是 Q2，添加两阶段模型指标和稳定性选择信息
        if two_stage_metrics is not None:
            final_metrics["two_stage"] = two_stage_metrics
        if stability_info is not None:
            final_metrics["stability_selection"] = stability_info
            
        print("最终选定特征列表：")
        for i, f in enumerate(final_feats):
            print(f"  {i+1}. {f}")
        
        # 保存结果
        # 清理 target_info 中的 warnings (它是个列表，需要保留)
        target_info_safe = {k: v for k, v in target_info.items() if k != 'warnings'}
        target_info_safe['warnings'] = target_info.get('warnings', [])
        target_info_safe['is_problematic'] = is_problematic

        result = {
            "target": target_col,
            "initial_count": X.shape[1],
            "after_roi_crop_count": X.shape[1],
            "after_corr_count": X_uncorr_df.shape[1],
            "dropped_roi": existing_drop_cols,
            "dropped_correlation": dropped_corr,
            "selected_features": final_feats,
            "selection_method": final_metrics["method"],
            "cv_metrics": final_metrics,
            "eval_model": "TwoStage_Stability" if q == 2 else "RidgeCV_Pipeline_Stratified",
            "target_distribution": target_info_safe
        }

        # Q2 使用稳定性选择，其他问题使用 ElasticNet
        if q == 2:
            result["stability_selection_params"] = {
                "n_bootstrap": 50,
                "sample_ratio": 0.7,
                "threshold": 0.5,
                "cv": "StratifiedKFold(n_splits=5, by_group)"
            }
        else:
            result["elasticnet_params"] = {
                "alpha": float(best_alpha),
                "l1_ratio": float(best_l1_ratio),
                "cv": "StratifiedKFold(n_splits=5, by_group)"
            }
        
        out_file = os.path.join(output_dir, f"selected_features_q{q}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"筛选结果已保存至 {out_file}")

    print("\n" + "="*50)
    print("回归特征分析完成。")
    
if __name__ == "__main__":
    main()
