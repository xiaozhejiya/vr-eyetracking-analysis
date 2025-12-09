# analysis 目录说明

本目录包含眼动事件分析、RQA 指标计算，以及为建模准备的事件与特征导出脚本。

## 数据目录约定
- 预处理数据根：`data/data_processed/<group>_processed/<subject>/..._preprocessed.csv`
- 校准数据根（三选一）：
  - 网格：`data/data_calibration/<group>_calibrated/<subject>/..._preprocessed_calibrated.csv`
  - 梯度：`data/data_calibration_gradient_ascent/<group>_calibrated/...`
  - 混合：`data/data_calibration_mix/<group>_calibrated/...`
- 事件导出：`data/MLP_data/event_data/<group>_group/<group>_q{1..5}.csv`
- 事件聚合特征：`data/MLP_data/features/event_features/<group>_group/<group>_q{1..5}.csv`

## 核心模块
- `analysis/event_analyzer.py`
  - 眼动事件识别（IVT）与 ROI 统计。
  - 主要函数：
    - `process_single_file(path, debug=False)`，返回 `(evt_df, roi_df)`。
    - ROI 定义获取：`get_roi_def(adq_id)`。
  - 参考：速度计算在 `analysis/event_analyzer.py:149`；ROI 映射在 `analysis/event_analyzer.py:126`。
- `analysis/rqa_analyzer.py` 与 `analysis/rqa_fast.py`
  - 计算一维/二维的 RQA 指标（RR、DET、ENT 等）。
  - 可直接对单文件或批量文件计算并返回字典或 DataFrame。

## 事件导出（按组×题号聚合到 CSV）
- 脚本：`analysis/export_events_for_mlp.py`
- 用途：遍历指定校准源目录，按组与题号（q1–q5）导出事件数据到 `event_data`。
- 用法：
  - `python analysis/export_events_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix`
  - `--calibrated-dir-name` 可选：`data_calibration`、`data_calibration_mix`、`data_calibration_gradient_ascent`。
- 输出示例：`data/MLP_data/event_data/ad_group/ad_q1.csv`，每行一个事件，包含 `EventType`、`Duration_ms`、`Amplitude_deg`、`MaxVel`、`MeanVel`、`ROI` 等。

## 事件特征聚合（受试者级）
- 脚本：`analysis/aggregate_mlp_features.py`
- 用途：将 `event_data` 中的事件按 `subject` 聚合为受试者级特征（每题一行）。
- 用法：
  - `python analysis/aggregate_mlp_features.py --groups control,mci,ad`
- 输出：`data/MLP_data/features/event_features/<group>_group/<group>_q{1..5}.csv`
  - 每行一个受试者在该题的特征，包含：
    - 计数与时长：`fixation_count`、`saccade_count`、`fixation_duration_ms`、`saccade_duration_ms`
    - 幅度与速度：`fixation_amplitude_mean_deg`、`saccade_amplitude_mean_deg`、`saccade_maxvel_mean`、`saccade_meanvel_mean`
    - ROI 占比：`kw_fix_duration_ms`、`inst_fix_duration_ms`、`bg_fix_duration_ms`、`kw_time_ratio`、`inst_time_ratio`、`bg_time_ratio`

## RQA 特征导出（受试者级）
- 脚本：`analysis/export_rqa_features_for_mlp.py`
- 用途：遍历指定校准源目录，计算每个受试者每题的 RQA 指标（`RR`、`DET`、`ENT`），并按 `q1–q5` 输出受试者级特征表。
- 参数：
  - `--calibrated-dir-name` 选择校准目录：`data_calibration` | `data_calibration_mix` | `data_calibration_gradient_ascent`
  - `--m --tau --eps --lmin` 使用 `config/settings.py` 中的默认值，并做范围裁剪。
- 用法：
  - `python analysis/export_rqa_features_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix`
- 输出：`data/MLP_data/features/rqa_features/<group>_group/<group>_q{1..5}.csv`
  - 每行一个受试者在该题的 RQA 指标，若同一受试者同一题号有多条记录会聚合为均值。

## 合并事件与 RQA 特征
- 脚本：`analysis/merge_event_rqa_features.py`
- 用途：将事件特征与 RQA 特征按键 `group, subject, q` 合并，得到统一的受试者级特征表。
- 参数：
  - `--event-dir-name` 事件特征目录名，默认 `event_features`
  - `--rqa-dir-name` RQA 特征目录名，默认 `rqa_features`
  - `--out-dir-name` 合并输出目录名，默认 `merged_features`
  - `--join` 合并方式：`inner`（默认）或 `outer`
- 用法：
  - `python analysis/merge_event_rqa_features.py --groups control,mci,ad --join inner`
- 输出：`data/MLP_data/features/merged_features/<group>_group/<group>_q{1..5}.csv`
  - 列包含事件特征与 RQA 指标，适合直接用于模型训练。

## 推荐工作流
1. 选择校准源并导出事件：
   - `python analysis/export_events_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix`
2. 对事件做受试者级聚合：
   - `python analysis/aggregate_mlp_features.py --groups control,mci,ad`
3. 导出 RQA 特征：
   - `python analysis/export_rqa_features_for_mlp.py --groups control,mci,ad --calibrated-dir-name data_calibration_mix`
4. 合并事件与 RQA 特征为最终训练集：
   - `python analysis/merge_event_rqa_features.py --groups control,mci,ad --join inner`

## 约定与说明
- 组名：`control`、`mci`、`ad`，对应目录名 `<group>_processed` / `<group>_calibrated`。
- 题号：`q1`–`q5`，由文件名解析（如 `ad10q1_preprocessed_calibrated.csv` -> `q1`）。
- 项目根：脚本内部使用 `project_root()` 指向 `lsh_eye_analysis` 根目录，并在其下定位 `data/`。

