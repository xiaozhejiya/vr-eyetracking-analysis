# 通用工具模块

该目录包含项目中使用的通用工具函数和辅助类。

## 文件说明

- **`score_function.py`**
  核心评分与计算模块。提供了用于校准的评分函数，包括：
  - `calculate_score_grid`: 执行网格搜索以寻找最佳偏移。
  - `calculate_score_and_metrics`: 计算给定偏移下的最终得分及详细指标（如注视时间比例）。
  - `apply_offset`: 对眼动数据应用偏移校准。
  - 支持硬 ROI（矩形区域）和软 ROI（概率分布）两种计算模式。

- **`CosineWarmupDecay.py`**
  自定义的学习率调度器。实现了带预热（Warmup）的余弦退火衰减策略，用于 PyTorch 优化器。在梯度上升校准过程中动态调整学习率，以实现更稳定和精确的收敛。

- **`test_gpu.py`**
  简单的测试脚本。用于检查 PyTorch 是否能正确识别并使用 CUDA GPU 加速，确保计算环境配置正确。