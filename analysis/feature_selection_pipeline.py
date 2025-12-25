import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
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


def evaluate_subset_regressor(X, y, feature_names, model_type='rf', cv=None):
    """
    使用交叉验证评估回归效果 (R^2 Score 和 MAE)
    """
    if not feature_names:
        print("  [警告] 特征列表为空，跳过评估")
        return np.nan, np.nan
        
    X_subset = X[feature_names]
    
    # 交叉验证策略
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 构建 Pipeline 以防止数据泄露
    # 大多数回归模型需要标准化，RF 不需要但也不受害
    steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]
    
    if model_type == 'rf':
        # RF 不需要 StandardScaler，可以只用 Imputer
        steps = [('imputer', SimpleImputer(strategy='mean'))]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'lasso':
        model = LassoCV(cv=5, random_state=42, max_iter=10000)
    elif model_type == 'ridge':
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    pipeline = Pipeline(steps + [('model', model)])
    
    # 计算 R2
    r2_scores = cross_val_score(pipeline, X_subset, y, cv=cv, scoring='r2')
    # 计算 MAE (Negative MAE, so flip sign)
    mae_scores = cross_val_score(pipeline, X_subset, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return r2_scores.mean(), -mae_scores.mean()

def main():
    output_dir = os.path.join(project_root(), "data", "MLP_data", "selected_features_regression")
    os.makedirs(output_dir, exist_ok=True)
    
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
                print(f"  ⚠️  {warn}")
        
        # --- 2.2 构建分层 CV (Stratified CV) ---
        # 从 subject ID 中提取 group (例如 'ad_group_1' -> 'ad')
        # 用于 StratifiedKFold 保证各 fold 中组别比例一致
        group_labels = df['subject'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'unknown')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(skf.split(X, group_labels))
        
        # --- 2.3 数据预处理 (用于特征筛选步骤) ---
        # 注意：这里的 imputed/scaled 数据仅用于计算相关性和 RF/Lasso 筛选特征
        # 最终评估时会使用 Pipeline 在 CV 内部重新处理，避免泄露
        
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
        
        # 统一使用 RidgeCV + Pipeline + StratifiedCV 评估
        r2, mae = evaluate_subset_regressor(X, y, enet_feats_top, model_type='ridge', cv=cv_splits)

        # 计算基线（DummyRegressor 预测均值）的 MAE 作为参考
        dummy = DummyRegressor(strategy='mean')
        dummy_mae_scores = cross_val_score(dummy, X[enet_feats_top] if enet_feats_top else X.iloc[:, :1],
                                           y, cv=cv_splits, scoring='neg_mean_absolute_error')
        baseline_mae = -dummy_mae_scores.mean()

        r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        mae_display = f"{mae:.4f}" if not np.isnan(mae) else "N/A"
        print(f"  [统一评估] ElasticNet 特征集 (RidgeCV) -> R^2: {r2_display}, MAE: {mae_display}")
        print(f"  [基线参考] 预测均值的 MAE: {baseline_mae:.4f}")

        if not np.isnan(r2) and r2 < 0:
            print(f"  ⚠️  R² 为负数，模型表现不如预测均值，可能因为：")
            print(f"      - 目标变量方差过小或分布高度不均")
            print(f"      - 特征与目标之间缺乏线性关系")
            print(f"      - 样本量过小导致交叉验证不稳定")
        
        final_feats = enet_feats_top
        # 处理 NaN 值 (JSON 不支持 NaN)
        r2_safe = None if np.isnan(r2) else float(r2)
        mae_safe = None if np.isnan(mae) else float(mae)
        final_metrics = {"r2": r2_safe, "mae": mae_safe, "baseline_mae": float(baseline_mae), "method": "elasticnet"}
            
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
            "eval_model": "RidgeCV_Pipeline_Stratified",
            "elasticnet_params": {
                "alpha": float(best_alpha),
                "l1_ratio": float(best_l1_ratio),
                "cv": "StratifiedKFold(n_splits=5, by_group)"
            },
            "target_distribution": target_info_safe
        }
        
        out_file = os.path.join(output_dir, f"selected_features_q{q}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"筛选结果已保存至 {out_file}")

    print("\n" + "="*50)
    print("回归特征分析完成。")
    
if __name__ == "__main__":
    main()
