# VR Eye Tracking Analysis Project

本项目旨在分析 VR 环境下的眼动数据，通过预处理、空间校准、事件检测和非线性动力学分析（RQA），提取多维特征以区分正常对照组（Control）、轻度认知障碍组（MCI）和阿尔兹海默症组（AD）。

## 目录结构 (Directory Structure)

```
.
├── analysis/                                   # 特征提取与分析核心代码
│   ├── aggregate_mlp_features.py               # 聚合事件特征为受试者级数据
│   ├── export_events_for_mlp.py                # 导出详细眼动事件数据
│   ├── export_rqa_features_for_mlp.py          # 导出 RQA (递归图) 特征
│   ├── merge_event_rqa_features.py             # 合并事件特征与 RQA 特征
│   ├── rqa_analyzer.py                         # RQA 分析核心算法
│   └── event_analyzer.py                       # 眼动事件 (IVT) 与 ROI 分析核心算法
├── config/                                     # 项目配置
│   └── settings.py                             # 全局参数配置 (RQA参数, 路径等)
├── data/                                       # 数据存储目录
│   ├── data_raw/                               # 原始眼动数据 (TXT)
│   ├── data_processed/                         # 预处理后的 CSV 数据
│   ├── data_calibration*/                      # 校准后的数据 (多种校准方法输出)
    ├── MLP_data/                               # 机器学习用特征数据
    ├── MMSE_Score/                             # 受试者认知评分数据 (MMSE)
    └── background_images/                      # 刺激材料背景图
├── data_calibration_and_data_calibration_analysis/ # 数据校准脚本
│   ├── data_calibration.py                     # 网格搜索校准
│   ├── data_calibration_gradient_ascent.py     # 梯度上升校准
│   └── grid_search_mix_gradient_ascent.py      # 混合校准策略
├── data_processing/                            # 原始数据预处理
│   └── vr_eyetracking_processor.py             # TXT 转 CSV 处理器
├── utils/                                      # 通用工具
│   ├── db_viewer.py                            # 数据库连接与查看工具
│   ├── score_function.py                       # 校准评分函数 (核心算法)
│   └── CosineWarmupDecay.py                    # 学习率调度器 (用于梯度上升)
└── Visualization/                              # 可视化脚本
    ├── visualize_roi_and_trajectory.py         # 轨迹与 ROI 可视化
    └── visualize_data_calibartion.py           # 校准效果可视化
```

## 环境要求 (Requirements)

请确保安装以下 Python 库：
```bash
pip install numpy pandas scipy matplotlib seaborn pillow sqlalchemy pymysql torch tqdm
```

## 使用流程 (Usage Workflow)

### 1. 数据预处理 (Preprocessing)
将原始设备导出的 TXT 数据转换为标准 CSV 格式。
脚本位置：`data_processing/vr_eyetracking_processor.py`

### 2. 数据校准 (Calibration)
修正 VR 头显佩戴导致的眼动数据整体偏移。
提供三种校准策略，脚本位于 `data_calibration_and_data_calibration_analysis/`：
- **网格搜索 (`data_calibration.py`)**: 全局搜索最优偏移量。
- **梯度上升 (`data_calibration_gradient_ascent.py`)**: 基于软 ROI 概率图优化。
- **混合策略 (`grid_search_mix_gradient_ascent.py`)**: 先网格搜索后梯度微调（推荐）。

运行示例：
```bash
# 使用混合策略校准所有组
python data_calibration_and_data_calibration_analysis/grid_search_mix_gradient_ascent.py --groups control,mci,ad
```

### 3. 特征提取 pipeline (Feature Extraction)

特征提取分为三个步骤，最终生成用于 MLP/机器学习模型的特征表。

#### 步骤 3.1: 导出眼动事件
基于 IVT (Velocity-Threshold Identification) 算法识别注视 (Fixation) 和扫视 (Saccade)。
```bash
# 指定使用混合校准后的数据
python analysis/export_events_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix
```
此步骤会在 `data/MLP_data/event_data/` 生成详细事件列表。

#### 步骤 3.2: 聚合事件特征
将详细事件列表聚合为受试者级别的统计特征（如平均注视时长、ROI 停留占比等）。
```bash
python analysis/aggregate_mlp_features.py --groups control,mci,ad
```
输出位于：`data/MLP_data/features/event_features/`

#### 步骤 3.3: 提取 RQA 特征
计算眼动轨迹的递归量化分析指标 (RR, DET, ENT, L_max 等)。
参数在 `config/settings.py` 中配置。
```bash
python analysis/export_rqa_features_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix
```
输出位于：`data/MLP_data/features/rqa_features/`

#### 步骤 3.4: 合并所有特征
将事件特征与 RQA 特征合并为一个宽表，每行代表一个受试者在某道题上的完整特征向量。
```bash
python analysis/merge_event_rqa_features.py --groups control,mci,ad --join inner
```
最终文件位于：`data/MLP_data/features/merged_features/`，可直接用于模型训练。

### 4. 数据库工具 (Database Tools)
用于连接和检查远程 MySQL 数据库中的眼动数据。
脚本：`utils/db_viewer.py`

```bash
# 查看数据库中的表
python utils/db_viewer.py --uri "mysql+pymysql://user:password@host/dbname" --action tables
```

### 5. 可视化 (Visualization)
`Visualization/` 目录下包含多种可视化脚本，用于生成热力图、轨迹图以及校准前后的对比图。

## 配置 (Configuration)
项目的主要配置位于 `config/settings.py`，包括：
- RQA 分析参数 (`dim`, `tau`, `epsilon` 等)
- 预处理阈值 (速度阈值、滤波参数)
- 数据路径常量
