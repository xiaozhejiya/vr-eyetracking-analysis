"""
可视化 Q1–Q5 背景图的 ROI，并叠加已校准的眼动轨迹
- 通过常量 `GROUP_TYPE` 指定组别，从 `lsh_eye_analysis/data_calibartion/{group}_calibrated/{subject_folder}` 读取数据
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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def calibration_dir(*parts):
    return os.path.join(project_root(), "lsh_eye_analysis", "data_calibartion", *parts)

def data_dir(*parts):
    return os.path.join(project_root(), "data", *parts)

def ensure_output_dir():
    out = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out, exist_ok=True)
    return out

def import_event_analyzer():
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()

def load_background_image(q_num):
    p = data_dir("background_images", f"Q{q_num}.jpg")
    if not os.path.exists(p):
        return None
    return Image.open(p).convert("RGB")

def draw_roi_layer(base_img, roi_kw, roi_inst, roi_bg):
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
            y_top = int((1 - ymn) * h)
            y_bot = int((1 - ymy) * h)
            y1, y2 = min(y_top, y_bot), max(y_top, y_bot)
            d.rectangle([(x1, y1), (x2, y2)], fill=(color[0], color[1], color[2], alpha), outline=(color[0], color[1], color[2], 255), width=2)
            tw, th = d.textlength(rn, font=font_), font_.size
            tx = min(x1 + 2, w - tw)
            ty = min(y2 + 2, h - th)
            d.rectangle([(tx, ty), (tx + tw, ty + th)], fill=(255, 255, 255, 160))
            d.text((tx, ty), rn, fill=(0, 0, 0, 255), font=font_)
    draw_rois(roi_bg, (0, 128, 255), 60)
    draw_rois(roi_inst, (255, 165, 0), 80)
    draw_rois(roi_kw, (255, 0, 0), 100)
    return layer

def draw_trajectory_on_image(base_img, df, point_size=1, line_width=2):
    w, h = base_img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    pts = [(int(xs[i] * w), int((1 - ys[i]) * h)) for i in range(len(xs))]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=(50, 200, 120, 160), width=line_width)
    for x, y in pts:
        d.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill=(0, 0, 255, 160))
    if pts:
        sx, sy = pts[0]
        d.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), fill=(0, 255, 0, 220))
        ex, ey = pts[-1]
        d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(255, 0, 0, 220))
    return layer

def visualize_five_backgrounds():
    analyzer = import_event_analyzer()
    out_dir = ensure_output_dir()
    results = []
    for q in [1, 2, 3, 4, 5]:
        img = load_background_image(q)
        if img is None:
            continue
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
        roi_layer = draw_roi_layer(img, kw, inst, bg)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer).convert("RGB")
        out_path = os.path.join(out_dir, f"background_Q{q}_roi.jpg")
        combined.save(out_path)
        results.append(out_path)
    return results

def parse_q_num_from_filename(filename):
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1

def pick_random_calibrated_file(group_type):
    root = calibration_dir(f"{group_type}_calibrated")
    if not os.path.exists(root):
        return None
    subject_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subject_dirs:
        return None
    sd = random.choice(subject_dirs)
    files = [f for f in os.listdir(sd) if f.endswith("_preprocessed_calibrated.csv")]
    if not files:
        return None
    return os.path.join(sd, random.choice(files))

def list_subject_q_files_calibrated(folder):
    files = {}
    for name in os.listdir(folder):
        if name.endswith("_preprocessed_calibrated.csv"):
            m = re.search(r"q(\d)", name.lower())
            if m:
                q = int(m.group(1))
                files[q] = os.path.join(folder, name)
    return files

def pick_random_subject_folder_calibrated(group_type, require_complete=True):
    root = calibration_dir(f"{group_type}_calibrated")
    if not os.path.exists(root):
        return None
    candidates = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not candidates:
        return None
    random.shuffle(candidates)
    if require_complete:
        for folder in candidates:
            q_files = list_subject_q_files_calibrated(folder)
            if all(q in q_files for q in [1,2,3,4,5]):
                return folder
    return candidates[0]

def visualize_subject_all_questions_calibrated(group_type):
    folder = pick_random_subject_folder_calibrated(group_type, require_complete=True)
    if not folder:
        return []
    q_files = list_subject_q_files_calibrated(folder)
    out_dir = ensure_output_dir()
    analyzer = import_event_analyzer()
    results = []
    for q in [1,2,3,4,5]:
        fp = q_files.get(q)
        if not fp:
            continue
        df = pd.read_csv(fp)
        if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
            continue
        img = load_background_image(q)
        if img is None:
            continue
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
        roi_layer = draw_roi_layer(img, kw, inst, bg)
        traj_layer = draw_trajectory_on_image(img, df)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)
        combined = Image.alpha_composite(combined, traj_layer).convert("RGB")
        out_name = f"calibrated_trajectory_{group_type}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
        out_path = os.path.join(out_dir, out_name)
        combined.save(out_path)
        results.append(out_path)
    return results

def visualize_random_calibrated(group_type):
    fp = pick_random_calibrated_file(group_type)
    if not fp:
        return None
    df = pd.read_csv(fp)
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return None
    q = parse_q_num_from_filename(os.path.basename(fp))
    img = load_background_image(q)
    if img is None:
        return None
    analyzer = import_event_analyzer()
    kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
    roi_layer = draw_roi_layer(img, kw, inst, bg)
    traj_layer = draw_trajectory_on_image(img, df)
    combined = Image.alpha_composite(img.convert("RGBA"), roi_layer)
    combined = Image.alpha_composite(combined, traj_layer).convert("RGB")
    out_dir = ensure_output_dir()
    out_name = f"calibrated_trajectory_{GROUP_TYPE}_Q{q}_{os.path.basename(fp).replace('.csv','')}.png"
    out_path = os.path.join(out_dir, out_name)
    combined.save(out_path)
    return out_path

def main():
    bg_paths = visualize_five_backgrounds()
    traj_paths = visualize_subject_all_questions_calibrated(GROUP_TYPE)
    print("ROI backgrounds:", bg_paths)
    print("Calibrated subject trajectories:", traj_paths)

if __name__ == "__main__":
    main()