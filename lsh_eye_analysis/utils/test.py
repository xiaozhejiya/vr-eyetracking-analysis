"""
基于阅读模式的眼动数据校准
利用阅读的自然规律（从左到右，从上到下）来校准系统性偏差
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


class ReadingBasedCalibrator:
    """基于阅读行为模式的眼动校准器"""

    def __init__(self, text_roi):
        """
        Parameters:
        -----------
        text_roi : tuple
            文本区域 (x_min, y_min, x_max, y_max)
        """
        self.text_roi = text_roi
        self.dx = 0.0
        self.dy = 0.0

    def detect_fixations(self, df, velocity_threshold=0.05, min_duration=100):
        """
        检测注视点（使用速度阈值法）

        Parameters:
        -----------
        df : DataFrame
            包含 x, y, milliseconds 列的数据
        velocity_threshold : float
            速度阈值（归一化坐标）
        min_duration : float
            最小注视时长（毫秒）

        Returns:
        --------
        fixations : list of dict
            每个注视点包含 {x, y, start_time, duration}
        """
        if len(df) < 2:
            return []

        xs = df['x'].values
        ys = df['y'].values
        times = df['milliseconds'].values

        # 计算速度
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])
        dt = np.diff(times, prepend=times[0])
        dt = np.maximum(dt, 1)  # 避免除零

        velocity = np.sqrt(dx ** 2 + dy ** 2) / (dt / 1000.0)  # 单位/秒

        # 注视点标记（速度低于阈值）
        is_fixation = velocity < velocity_threshold

        # 聚合连续的注视点
        fixations = []
        i = 0
        while i < len(is_fixation):
            if is_fixation[i]:
                start = i
                while i < len(is_fixation) and is_fixation[i]:
                    i += 1
                end = i

                duration = times[end - 1] - times[start]
                if duration >= min_duration:
                    fixations.append({
                        'x': np.median(xs[start:end]),
                        'y': np.median(ys[start:end]),
                        'start_time': times[start],
                        'duration': duration,
                        'start_idx': start,
                        'end_idx': end
                    })
            else:
                i += 1

        return fixations

    def cluster_fixations_into_lines(self, fixations, y_tolerance=0.05):
        """
        将注视点聚类成阅读行

        Parameters:
        -----------
        fixations : list of dict
            注视点列表
        y_tolerance : float
            y坐标容差（用于判断同一行）

        Returns:
        --------
        lines : list of list
            每一行的注视点列表（按时间排序）
        """
        if not fixations:
            return []

        # 按时间排序
        fixations_sorted = sorted(fixations, key=lambda f: f['start_time'])

        # 使用DBSCAN在y方向聚类
        y_coords = np.array([f['y'] for f in fixations_sorted]).reshape(-1, 1)
        clustering = DBSCAN(eps=y_tolerance, min_samples=2).fit(y_coords)

        # 组织成行
        lines = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # 噪声点
                continue
            if label not in lines:
                lines[label] = []
            lines[label].append(fixations_sorted[i])

        # 按平均y坐标排序行（从上到下）
        sorted_lines = sorted(lines.values(),
                              key=lambda line: np.mean([f['y'] for f in line]))

        return sorted_lines

    def calculate_reading_score(self, fixations, dx, dy):
        """
        计算给定偏移量下的"阅读合理性"得分

        考虑因素：
        1. 从左到右的扫视占比（正向扫视）
        2. 行与行之间向下移动
        3. 注视点在文本ROI内的比例
        4. 扫视距离的合理性
        """
        if not fixations:
            return -np.inf

        # 应用偏移
        adj_fixations = []
        for f in fixations:
            adj_x = np.clip(f['x'] + dx, 0, 1)
            adj_y = np.clip(f['y'] + dy, 0, 1)
            adj_fixations.append({**f, 'x': adj_x, 'y': adj_y})

        # 聚类成行
        lines = self.cluster_fixations_into_lines(adj_fixations)

        if len(lines) == 0:
            return -np.inf

        score = 0.0

        # 1. 行内从左到右的扫视比例
        forward_saccades = 0
        backward_saccades = 0
        for line in lines:
            if len(line) < 2:
                continue
            for i in range(len(line) - 1):
                dx_saccade = line[i + 1]['x'] - line[i]['x']
                if dx_saccade > 0.01:  # 向右
                    forward_saccades += 1
                elif dx_saccade < -0.01:  # 向左（回退）
                    backward_saccades += 1

        if forward_saccades + backward_saccades > 0:
            forward_ratio = forward_saccades / (forward_saccades + backward_saccades)
            score += forward_ratio * 100  # 权重100

        # 2. 行间向下移动
        line_transitions = 0
        downward_transitions = 0
        for i in range(len(lines) - 1):
            if len(lines[i]) == 0 or len(lines[i + 1]) == 0:
                continue
            # 比较相邻行的y坐标
            y1 = np.mean([f['y'] for f in lines[i]])
            y2 = np.mean([f['y'] for f in lines[i + 1]])
            line_transitions += 1
            if y2 > y1:  # 向下
                downward_transitions += 1

        if line_transitions > 0:
            down_ratio = downward_transitions / line_transitions
            score += down_ratio * 50  # 权重50

        # 3. ROI内的注视点比例
        x_min, y_min, x_max, y_max = self.text_roi
        inside_roi = sum(1 for f in adj_fixations
                         if x_min <= f['x'] <= x_max and y_min <= f['y'] <= y_max)
        roi_ratio = inside_roi / len(adj_fixations)
        score += roi_ratio * 150  # 权重150

        # 4. 扫视距离合理性（中文阅读通常2-8个字符）
        # 假设文本宽度对应字符数
        reasonable_saccades = 0
        total_saccades = 0
        for line in lines:
            if len(line) < 2:
                continue
            for i in range(len(line) - 1):
                dx_saccade = abs(line[i + 1]['x'] - line[i]['x'])
                # 假设0.02-0.15为合理范围
                if 0.02 <= dx_saccade <= 0.15:
                    reasonable_saccades += 1
                total_saccades += 1

        if total_saccades > 0:
            reasonable_ratio = reasonable_saccades / total_saccades
            score += reasonable_ratio * 50  # 权重50

        # 5. 惩罚过大的跳跃（可能是噪声）
        large_jumps = 0
        for i in range(len(adj_fixations) - 1):
            dist = np.sqrt((adj_fixations[i + 1]['x'] - adj_fixations[i]['x']) ** 2 +
                           (adj_fixations[i + 1]['y'] - adj_fixations[i]['y']) ** 2)
            if dist > 0.3:  # 超过30%的屏幕
                large_jumps += 1

        score -= large_jumps * 10  # 惩罚

        return score

    def calibrate(self, df, dx_range=(-0.15, 0.15), dy_range=(-0.15, 0.15),
                  grid_step=0.01, refine_step=0.002):
        """
        两阶段校准：粗搜索 + 精细优化

        Parameters:
        -----------
        df : DataFrame
            原始眼动数据
        dx_range, dy_range : tuple
            搜索范围
        grid_step : float
            粗搜索步长
        refine_step : float
            精细搜索步长

        Returns:
        --------
        result : dict
            包含最优偏移量和评分
        """
        # 先检测注视点
        fixations = self.detect_fixations(df)

        if len(fixations) < 5:
            print("Warning: Too few fixations detected, calibration may be unreliable")
            return {'dx': 0, 'dy': 0, 'score': -np.inf, 'fixations': fixations}

        print(f"Detected {len(fixations)} fixations")

        # 阶段1: 粗网格搜索
        print("Stage 1: Coarse grid search...")
        dx_vals = np.arange(dx_range[0], dx_range[1] + grid_step / 2, grid_step)
        dy_vals = np.arange(dy_range[0], dy_range[1] + grid_step / 2, grid_step)

        best_score = -np.inf
        best_dx = 0
        best_dy = 0

        for dx in dx_vals:
            for dy in dy_vals:
                score = self.calculate_reading_score(fixations, dx, dy)
                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy

        print(f"Coarse result: dx={best_dx:.4f}, dy={best_dy:.4f}, score={best_score:.2f}")

        # 阶段2: 精细优化
        print("Stage 2: Fine-grained optimization...")

        def objective(params):
            return -self.calculate_reading_score(fixations, params[0], params[1])

        result = minimize(
            objective,
            x0=[best_dx, best_dy],
            method='Powell',
            options={'xtol': refine_step, 'ftol': 0.01}
        )

        self.dx = result.x[0]
        self.dy = result.x[1]
        final_score = -result.fun

        print(f"Final result: dx={self.dx:.4f}, dy={self.dy:.4f}, score={final_score:.2f}")

        return {
            'dx': float(self.dx),
            'dy': float(self.dy),
            'score': float(final_score),
            'fixations': fixations,
            'coarse_dx': float(best_dx),
            'coarse_dy': float(best_dy),
            'coarse_score': float(best_score)
        }

    def apply(self, df):
        """应用校准偏移"""
        result = df.copy()
        if 'x' in result.columns:
            result['x'] = np.clip(result['x'] + self.dx, 0, 1)
        if 'y' in result.columns:
            result['y'] = np.clip(result['y'] + self.dy, 0, 1)
        return result


# 与你原有代码接口兼容的函数
def calibrate_file_reading_based(file_path, text_roi, apply=False, save_path=None):
    """
    使用阅读模式校准单个文件

    Parameters:
    -----------
    file_path : str
        输入CSV文件路径
    text_roi : tuple
        文本ROI (x_min, y_min, x_max, y_max)
    apply : bool
        是否保存校准后的文件
    save_path : str
        输出路径（如果为None则自动生成）

    Returns:
    --------
    result : dict
        校准结果
    """
    df = pd.read_csv(file_path)

    calibrator = ReadingBasedCalibrator(text_roi)
    calib_result = calibrator.calibrate(df)

    applied_path = None
    if apply:
        calibrated_df = calibrator.apply(df)
        if save_path is None:
            save_path = file_path.replace('.csv', '_calibrated.csv')
        calibrated_df.to_csv(save_path, index=False)
        applied_path = save_path
        print(f"Saved calibrated data to: {save_path}")

    return {
        'file_path': file_path,
        'calibration': calib_result,
        'applied_path': applied_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reading-based eye tracking calibration")
    parser.add_argument("--file", type=str, required=True, help="CSV file path")
    parser.add_argument("--roi", type=str, required=True,
                        help="Text ROI as 'x_min,y_min,x_max,y_max'")
    parser.add_argument("--apply", action="store_true", help="Save calibrated data")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    # 解析ROI
    roi_parts = [float(x) for x in args.roi.split(',')]
    if len(roi_parts) != 4:
        raise ValueError("ROI must be 4 values: x_min,y_min,x_max,y_max")

    result = calibrate_file_reading_based(
        args.file,
        text_roi=tuple(roi_parts),
        apply=args.apply,
        save_path=args.output
    )

    print("\n=== Calibration Result ===")
    print(f"dx: {result['calibration']['dx']:.4f}")
    print(f"dy: {result['calibration']['dy']:.4f}")
    print(f"Score: {result['calibration']['score']:.2f}")
    if result['applied_path']:
        print(f"Saved to: {result['applied_path']}")