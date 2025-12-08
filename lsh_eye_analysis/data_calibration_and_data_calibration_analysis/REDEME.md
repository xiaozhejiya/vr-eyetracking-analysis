# 数据校准与分析

该目录包含用于眼动数据校准和分析的脚本及结果文件。

## 脚本文件

- **`data_calibration.py`**
  基于网格搜索（Grid Search）的校准方法。在指定的偏移范围内搜索最优的 $(dx, dy)$，使眼动轨迹与 ROI 的重合度得分最高。支持硬 ROI（矩形）和软 ROI（概率图）评分。

- **`data_calibration_gradient_ascent.py`**
  基于梯度上升（Gradient Ascent）的校准方法。使用 PyTorch 自动微分，针对软 ROI 评分函数直接优化 $(dx, dy)$。

- **`grid_search_mix_gradient_ascent.py`**
  混合校准方法。先使用网格搜索找到全局最优的初始点，再以此为起点进行梯度上升微调。结合了网格搜索的全局性和梯度上升的精确性，并引入了余弦退火学习率调度和早停机制。

## 结果文件

- **`grid_search_summary.csv`**
  `data_calibration.py` 运行后的校准结果汇总（评分和处理时间）。

- **`gradient_ascent_summary.csv`**
  `data_calibration_gradient_ascent.py` 运行后的校准结果汇总。

- **`mix_summary.csv`**
  `grid_search_mix_gradient_ascent.py` 运行后的校准结果汇总。
