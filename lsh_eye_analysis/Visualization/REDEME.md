# 可视化脚本

该目录包含用于可视化眼动数据、校准效果和 ROI（感兴趣区域）的脚本。

## 脚本文件

- **`visualize_data_calibartion.py`**
  用于可视化单个被试在校准前后的眼动轨迹对比。生成包含校准前和校准后轨迹的对比图，便于直观评估校准效果。

- **`visualize_data_calibartion_group.py`**
  批量可视化整个组（如 control, ad, mci）的校准结果。
  - 读取 `lsh_eye_analysis/data_calibration_grid_search_mix_gradient_ascent` 下的校准数据。
  - 生成每个被试所有题目的校准前后对比图。
  - 结果保存到 `Data_calibration_visualization_mix` 目录。
  - 可以通过常量DATA_CALIBRATION_DIR_NAME修改校准数据目录。
  - 可以通过常量VISUALIZATION_DIR_NAME修改可视化输出目录。

- **`visualize_roi_and_trajectory.py`**
  可视化硬 ROI（矩形区域）和眼动轨迹。将定义的 ROI 区域绘制在背景图上，并叠加眼动轨迹点，用于检查 ROI 定义是否准确以及轨迹分布情况。

- **`visualize_soft_roi_and_trajectory.py`**
  可视化软 ROI（概率图/热力图）和眼动轨迹。展示基于 Sigmoid 函数生成的软 ROI 概率分布，并叠加眼动轨迹，用于分析视线在渐变区域的分布。

## 输出目录

- **`Data_calibration_visualization/`**
  存放旧版或默认校准方法的可视化结果。

- **`Data_calibration_visualization_gradient_ascent/`**
  存放梯度上升校准方法的可视化结果。

- **`Data_calibration_visualization_mix/`**
  存放混合校准方法（网格搜索+梯度上升）的可视化结果。

- **`outputs/`**
  存放单次运行脚本生成的临时或示例可视化图片（如 ROI 示意图、单次轨迹图等）。