"""
可视化 Q1–Q5 背景图的 ROI，并叠加未校准的眼动轨迹。
- 通过常量 `GROUP_TYPE` 指定组别，从 `data/{group}_processed` 读取数据
- ROI 定义来自 `analysis.event_analyzer.EventAnalyzer`
- 输出文件保存到当前目录下的 `outputs`
"""
import os
import re
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

GROUP_TYPE = "control"

def project_root():
    """返回项目根目录的绝对路径，用于构造数据与模块的查找路径"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def data_dir(*parts):
    """拼接 `data/` 子目录的路径，例如 `data_dir('control_processed')`"""
    return os.path.join(project_root(), "data", *parts)

def load_background_image(q_num):
    """加载题目 Q{q_num} 的背景图，若不存在返回 `None`"""
    p = data_dir("background_images", f"Q{q_num}.jpg")
    if not os.path.exists(p):
        return None
    img = Image.open(p).convert("RGB")  # 背景图用 RGB，便于保存为 JPG
    return img

def ensure_output_dir():
    """确保 `outputs` 目录存在并返回其路径"""
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    return out

def import_event_analyzer():
    """动态导入 ROI 分析器（提供 Q1–Q5 的 ROI 定义）"""
    sys.path.append(project_root())  # 将项目根目录加入搜索路径
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()

# ======== 原来的“硬矩形 ROI”绘制函数（保留以便对比） ========

def draw_rect_roi_layer(base_img, roi_kw, roi_inst, roi_bg):
    """在透明图层上绘制三类 ROI（背景/说明/关键词）矩形与标签。
    输入 ROI 为归一化坐标 [0,1]，绘制时需将 y 轴翻转到图像坐标系。
    """
    w, h = base_img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    try:
        font_ = ImageFont.truetype("arial.ttf", 14)
    except:
        font_ = ImageFont.load_default()

    def draw_rois(roi_list, color, alpha):
        for rn, xmn, ymn, xmx, ymy in roi_list:
            x1 = int(xmn * w)
            x2 = int(xmx * w)
            y_top = int((1 - ymn) * h)  # 归一化坐标转像素并做 y 翻转
            y_bot = int((1 - ymy) * h)
            y1, y2 = min(y_top, y_bot), max(y_top, y_bot)
            d.rectangle(
                [(x1, y1), (x2, y2)],
                fill=(color[0], color[1], color[2], alpha),
                outline=(color[0], color[1], color[2], 255),
                width=2,
            )
            tw = d.textlength(rn, font=font_)
            th = font_.size
            tx = min(x1 + 2, w - tw)
            ty = min(y2 + 2, h - th)
            d.rectangle([(tx, ty), (tx + tw, ty + th)], fill=(255, 255, 255, 160))  # 标签底色
            d.text((tx, ty), rn, fill=(0, 0, 0, 255), font=font_)

    draw_rois(roi_bg,   (0, 128, 255), 60)
    draw_rois(roi_inst, (255, 165, 0), 80)
    draw_rois(roi_kw,   (255, 0, 0),   100)
    return layer

# ======== 新增：soft ROI 可视化（用 p_roi 显示颜色深度） ========

def _sigmoid_np(x):
    """数值稳定版 sigmoid，用于 soft ROI 计算。"""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def soft_rect_prob_grid(w, h, roi, k=50.0):
    """
    对一个矩形 ROI，在整张图上计算 soft membership：
        p(x,y) ≈ 1 表示在矩形内，≈0 表示在外部。
    返回 shape=(h, w) 的概率图（注意先 y 再 x）。
    """
    _, xmn, ymn, xmx, ymy = roi

    # 像素中心的归一化坐标：
    #   x_norm ∈ (0,1)，从左到右
    #   y_norm ∈ (0,1)，从下到上（和 ROI 的归一化坐标保持一致）
    xs = (np.arange(w) + 0.5) / float(w)         # (w,)
    ys_img = (np.arange(h) + 0.5) / float(h)     # 图像坐标，从上到下
    ys = 1.0 - ys_img                            # 归一化坐标，从下到上

    X, Y = np.meshgrid(xs, ys)  # both shape (h, w)

    px1 = _sigmoid_np(k * (X - xmn))
    px2 = _sigmoid_np(k * (xmx - X))
    py1 = _sigmoid_np(k * (Y - ymn))
    py2 = _sigmoid_np(k * (ymy - Y))

    return px1 * px2 * py1 * py2  # (h, w)

def build_soft_map_for_roi_list(w, h, roi_list, k=50.0):
    """
    对一组 ROI 叠加 soft membership：
      p_total(x,y) = sum_r p_r(x,y)，再 clip 到 [0,1]
    返回 shape=(h, w) 的概率图。
    """
    if not roi_list:
        return np.zeros((h, w), dtype=np.float32)
    p_total = np.zeros((h, w), dtype=np.float32)
    for roi in roi_list:
        p_total += soft_rect_prob_grid(w, h, roi, k=k)
    # 多个 ROI 叠加后做一个截断（防止 >1）
    p_total = np.clip(p_total, 0.0, 1.0)
    return p_total

def soft_map_to_rgba(map_, color, alpha_max):
    """
    将 [0,1] 的概率图 map_ 映射成 RGBA 图层：
      - 颜色固定为 color
      - alpha = map_ * alpha_max
    """
    h, w = map_.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (map_ * alpha_max).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")

def draw_soft_roi_layer(base_img, roi_kw, roi_inst, roi_bg, k=50.0):
    """
    使用 soft ROI 的方式绘制 ROI：
    - inst / kw: 仍然基于各自 ROI 矩形的 soft membership
    - 背景 bg: 不再使用 roi_bg，而是定义为 inst+kw 的补集：
        p_bg(x,y) = 1 - clip(p_inst(x,y) + p_kw(x,y), 0, 1)
      即：所有非 inst/kw 的区域都是“背景”，inst/kw 内部 p_bg≈0。
    """
    w, h = base_img.size

    # inst / kw 的 soft 概率图（“正向”区域）
    map_inst = build_soft_map_for_roi_list(w, h, roi_inst, k=k)
    map_kw   = build_soft_map_for_roi_list(w, h, roi_kw,   k=k)

    # ---- 关键改动：背景 = inst/kw 的补集 ----
    # 正向区域的总体覆盖概率
    map_pos = np.clip(map_inst + map_kw, 0.0, 1.0)
    # 背景 = 非 inst/kw 的地方
    map_bg  = 1.0 - map_pos

    # 分别映射为三张透明图层，再叠加
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # 你可以按需要调整 alpha_max，控制“热度”的强弱
    bg_layer   = soft_map_to_rgba(map_bg,   (0, 128, 255),  90)  # 蓝：背景（淡一点）
    inst_layer = soft_map_to_rgba(map_inst, (255, 165, 0), 160)  # 橙：指令
    kw_layer   = soft_map_to_rgba(map_kw,   (255, 0, 0),   200)  # 红：关键词

    # 叠加顺序：先背景，再指令，再关键词（关键词优先显示）
    layer = Image.alpha_composite(layer, bg_layer)
    layer = Image.alpha_composite(layer, inst_layer)
    layer = Image.alpha_composite(layer, kw_layer)

    return layer

# ======== 把 draw_roi_layer 换成 soft 版 ========

def visualize_five_backgrounds():
    """生成 Q1–Q5 的 soft ROI 叠加背景图并保存到 `outputs`"""
    analyzer = import_event_analyzer()
    out_dir = ensure_output_dir()
    results = []
    for q in [1, 2, 3, 4, 5]:
        img = load_background_image(q)
        if img is None:
            continue
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")  # 使用 n2q 映射获取 ROI 定义

        # 原来是 draw_rect_roi_layer，这里改成 soft 版（背景=inst+kw 的补集）
        roi_layer = draw_soft_roi_layer(img, kw, inst, bg, k=60.0)

        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer).convert("RGB")
        out_path = os.path.join(out_dir, f"background_Q{q}_soft_roi.jpg")
        combined.save(out_path)
        results.append(out_path)
    return results

def pick_random_processed_file(group_type):
    """从 `data/{group}_processed` 随机挑选一个未校准的 `_preprocessed.csv` 文件"""
    root = data_dir(f"{group_type}_processed")
    if not os.path.exists(root):
        return None
    group_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not group_folders:
        return None
    gf = random.choice(group_folders)
    files = [f for f in os.listdir(gf) if f.endswith("_preprocessed.csv")]
    if not files:
        return None
    return os.path.join(gf, random.choice(files))

def parse_q_num_from_filename(filename):
    """从文件名中解析题号 `q{digit}`，失败时默认返回 1"""
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1

def list_subject_q_files(folder):
    """列出受试者文件夹中的 Q1–Q5 预处理文件映射。
    返回 {q: filepath}，仅收集文件名包含 `_preprocessed.csv` 的条目。
    """
    files = {}
    for name in os.listdir(folder):
        if name.endswith("_preprocessed.csv"):
            m = re.search(r"q(\d)", name.lower())
            if m:
                q = int(m.group(1))
                files[q] = os.path.join(folder, name)
    return files

def pick_random_subject_folder(group_type, require_complete=True):
    """从指定组随机选择一个受试者文件夹。
    当 `require_complete=True` 时仅返回同时具备 Q1–Q5 的受试者。
    """
    root = data_dir(f"{group_type}_processed")
    if not os.path.exists(root):
        return None
    candidates = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not candidates:
        return None
    random.shuffle(candidates)  # 打乱顺序，保证随机性
    if require_complete:
        for folder in candidates:
            q_files = list_subject_q_files(folder)
            if all(q in q_files for q in [1,2,3,4,5]):
                return folder
    return candidates[0]

def draw_trajectory_on_image(base_img, df, point_size=1, line_width=2):
    """根据 DataFrame 的归一化坐标 `x,y` 绘制采样点与折线轨迹，并标注起止点。
    坐标转换：x_pix = x_norm*w；y_pix = (1 - y_norm)*h（图像坐标系的 y 向下）。
    """
    w, h = base_img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    pts = [(int(xs[i] * w), int((1 - ys[i]) * h)) for i in range(len(xs))]
    # 折线
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=(200, 80, 255, 160), width=line_width)
    # 采样点
    for x, y in pts:
        d.ellipse((x - point_size, y - point_size, x + point_size, y + point_size),
                  fill=(0, 0, 255, 160))
    # 起止点标记
    if pts:
        sx, sy = pts[0]
        d.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(0, 255, 0, 220))  # 起点
        ex, ey = pts[-1]
        d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(255, 0, 0, 220))  # 终点
    return layer

def visualize_subject_all_questions(group_type):
    """为选定组中随机选择的受试者，依次绘制 Q1–Q5 的 soft ROI + 轨迹叠加图。
    返回输出文件路径列表。仅处理含 `x,y` 列且非空的 CSV。
    """
    folder = pick_random_subject_folder(group_type, require_complete=True)
    if not folder:
        return []
    q_files = list_subject_q_files(folder)
    out_dir = ensure_output_dir()
    analyzer = import_event_analyzer()
    results = []
    for q in [1,2,3,4,5]:
        fp = q_files.get(q)
        if not fp:
            continue
        df = pd.read_csv(fp)  # 读取当前题目的未校准数据
        if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
            continue
        img = load_background_image(q)
        if img is None:
            continue
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
        roi_layer = draw_soft_roi_layer(img, kw, inst, bg, k=50.0)
        traj_layer = draw_trajectory_on_image(img, df)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)  # 先叠加 soft ROI
        combined = Image.alpha_composite(combined, traj_layer).convert("RGB")  # 再叠加轨迹
        out_name = f"trajectory_soft_{group_type}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
        out_path = os.path.join(out_dir, out_name)
        combined.save(out_path)
        results.append(out_path)
    return results

def visualize_random_uncalibrated(group_type):
    """随机抽取该组的一个未校准文件，绘制 soft ROI+轨迹，返回输出路径。"""
    fp = pick_random_processed_file(group_type)
    if not fp:
        return None
    df = pd.read_csv(fp)
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return None
    q = parse_q_num_from_filename(os.path.basename(fp))  # 从文件名解析题号
    img = load_background_image(q)
    if img is None:
        return None
    analyzer = import_event_analyzer()
    kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
    roi_layer = draw_soft_roi_layer(img, kw, inst, bg, k=50.0)
    traj_layer = draw_trajectory_on_image(img, df)
    combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)  # 先叠加 soft ROI
    combined = Image.alpha_composite(combined, traj_layer).convert("RGB")  # 再叠加轨迹
    out_dir = ensure_output_dir()
    out_name = f"trajectory_soft_{GROUP_TYPE}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
    out_path = os.path.join(out_dir, out_name)
    combined.save(out_path)
    return out_path

def main():
    """生成 Q1–Q5 的 soft ROI 背景图，并为同一受试者生成 Q1–Q5 的轨迹图。"""
    bg_paths = visualize_five_backgrounds()
    traj_paths = visualize_subject_all_questions(GROUP_TYPE)
    print("Soft ROI backgrounds:", bg_paths)
    print("Soft subject trajectories:", traj_paths)

if __name__ == "__main__":
    main()
