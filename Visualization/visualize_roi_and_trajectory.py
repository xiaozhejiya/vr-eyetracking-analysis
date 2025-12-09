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
# 临时指定受试者文件夹名称；置为 None 则随机
SUBJECT_FOLDER_NAME = "control_group_21"

def project_root():
    """返回项目根目录的绝对路径，用于构造数据与模块的查找路径"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def processed_dir(*parts):
    return os.path.join(project_root(), "data", "data_processed", *parts)

def assets_dir(*parts):
    return os.path.join(project_root(), "data", *parts)

def load_background_image(q_num):
    """加载题目 Q{q_num} 的背景图，若不存在返回 `None`"""
    p = assets_dir("background_images", f"Q{q_num}.jpg")
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

def draw_roi_layer(base_img, roi_kw, roi_inst, roi_bg):
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
            d.rectangle([(x1, y1), (x2, y2)], fill=(color[0], color[1], color[2], alpha), outline=(color[0], color[1], color[2], 255), width=2)
            tw, th = d.textlength(rn, font=font_), font_.size
            tx = min(x1 + 2, w - tw)
            ty = min(y2 + 2, h - th)
            d.rectangle([(tx, ty), (tx + tw, ty + th)], fill=(255, 255, 255, 160))  # 标签底色
            d.text((tx, ty), rn, fill=(0, 0, 0, 255), font=font_)
    draw_rois(roi_bg, (0, 128, 255), 60)
    draw_rois(roi_inst, (255, 165, 0), 80)
    draw_rois(roi_kw, (255, 0, 0), 100)
    return layer

def visualize_five_backgrounds():
    """生成 Q1–Q5 的 ROI 叠加背景图并保存到 `outputs`"""
    analyzer = import_event_analyzer()
    out_dir = ensure_output_dir()
    results = []
    for q in [1, 2, 3, 4, 5]:
        img = load_background_image(q)
        if img is None:
            continue
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")  # 使用 n2q 映射获取 ROI 定义
        roi_layer = draw_roi_layer(img, kw, inst, bg)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer).convert("RGB")
        out_path = os.path.join(out_dir, f"background_Q{q}_roi.jpg")
        combined.save(out_path)
        results.append(out_path)
    return results

def pick_random_processed_file(group_type):
    """从 `data/{group}_processed` 随机挑选一个未校准的 `_preprocessed.csv` 文件"""
    root = processed_dir(f"{group_type}_processed")
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
    root = processed_dir(f"{group_type}_processed")
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


def visualize_subject_all_questions(group_type):
    """为选定组中随机选择的受试者，依次绘制 Q1–Q5 的轨迹叠加 ROI。
    返回输出文件路径列表。仅处理含 `x,y` 列且非空的 CSV。
    """
    # folder = pick_random_subject_folder(group_type, require_complete=True)
    folder = os.path.join(processed_dir(f"{group_type}_processed"), str(SUBJECT_FOLDER_NAME)) if SUBJECT_FOLDER_NAME else pick_random_subject_folder(group_type, require_complete=True)
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
        roi_layer = draw_roi_layer(img, kw, inst, bg)
        traj_layer = draw_trajectory_on_image(img, df)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)  # 先叠加 ROI
        combined = Image.alpha_composite(combined, traj_layer).convert("RGB")  # 再叠加轨迹
        out_name = f"trajectory_{group_type}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
        out_path = os.path.join(out_dir, out_name)
        combined.save(out_path)
        results.append(out_path)
    return results

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
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=(200, 80, 255, 160), width=line_width)  # 轨迹折线
    for x, y in pts:
        d.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill=(0, 0, 255, 160))  # 采样点
    if pts:
        sx, sy = pts[0]
        d.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(0, 255, 0, 220))  # 起点
        ex, ey = pts[-1]
        d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(255, 0, 0, 220))  # 终点
    return layer

def visualize_random_uncalibrated(group_type):
    """随机抽取该组的一个未校准文件，绘制 ROI+轨迹，返回输出路径。"""
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
    roi_layer = draw_roi_layer(img, kw, inst, bg)
    traj_layer = draw_trajectory_on_image(img, df)
    combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)  # 先叠加 ROI
    combined = Image.alpha_composite(combined, traj_layer).convert("RGB")  # 再叠加轨迹
    out_dir = ensure_output_dir()
    out_name = f"trajectory_{GROUP_TYPE}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
    out_path = os.path.join(out_dir, out_name)
    combined.save(out_path)
    return out_path

def main():
    """生成 Q1–Q5 的 ROI 背景图，并为同一受试者生成 Q1–Q5 的轨迹图。"""
    bg_paths = visualize_five_backgrounds()
    traj_paths = visualize_subject_all_questions(GROUP_TYPE)
    print("ROI backgrounds:", bg_paths)
    print("Subject trajectories:", traj_paths)

if __name__ == "__main__":
    main()
