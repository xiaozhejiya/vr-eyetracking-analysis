"""
可视化 Q1–Q5 背景图的 ROI，并叠加已校准的眼动轨迹
- 通过常量 `GROUP_TYPE` 指定组别，从 `lsh_eye_analysis/data_calibration/{group}_calibrated/{subject_folder}` 读取数据
- ROI 定义来自 `analysis.event_analyzer.EventAnalyzer`
- 输出文件保存到当前目录下的 `outputs`
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle

GROUP_TYPES = ["control", "ad", "mci"]
DATA_CALIBRATION_DIR_NAME = "data_calibration"
VISUALIZATION_DIR_NAME = "Data_calibration_visualization"

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def calibration_dir(*parts):
    return os.path.join(project_root(), "data", DATA_CALIBRATION_DIR_NAME, *parts)

def data_dir(*parts):
    return os.path.join(project_root(), "data", *parts)

def visualization_root():
    return os.path.join(os.path.dirname(__file__), VISUALIZATION_DIR_NAME)

def import_event_analyzer():
    sys.path.append(project_root())
    from analysis.event_analyzer import EventAnalyzer
    return EventAnalyzer()

def load_background_image(q_num):
    p = data_dir("background_images", f"Q{q_num}.jpg")
    if not os.path.exists(p):
        return None
    import PIL.Image
    return PIL.Image.open(p).convert("RGB")

def draw_roi_on_axes(ax, w, h, roi_kw, roi_inst, roi_bg):
    def draw_rois(roi_list, color, alpha):
        for rn, xmn, ymn, xmx, ymy in roi_list:
            x1 = xmn * w
            x2 = xmx * w
            y_top = (1 - ymn) * h
            y_bot = (1 - ymy) * h
            y = min(y_top, y_bot)
            height = abs(y_top - y_bot)
            rect = Rectangle((x1, y), x2 - x1, height, linewidth=2,
                             edgecolor=(color[0]/255, color[1]/255, color[2]/255, 1.0),
                             facecolor=(color[0]/255, color[1]/255, color[2]/255, alpha/255))
            ax.add_patch(rect)
    draw_rois(roi_bg, (0, 128, 255), 60)
    draw_rois(roi_inst, (255, 165, 0), 80)
    draw_rois(roi_kw, (255, 0, 0), 100)

def draw_trajectory_on_axes(ax, w, h, df, point_size=8, line_width=2):
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    xp = xs * w
    yp = (1 - ys) * h
    ax.plot(xp, yp, color=(50/255, 200/255, 120/255, 0.7), linewidth=line_width)
    ax.scatter(xp, yp, s=point_size, color=(0, 0, 1, 0.6))
    if len(xp):
        ax.scatter([xp[0]], [yp[0]], s=30, color=(0, 1, 0, 0.9))
        ax.scatter([xp[-1]], [yp[-1]], s=30, color=(1, 0, 0, 0.9))

def list_subject_folders_calibrated(group_type):
    root = calibration_dir(f"{group_type}_calibrated")
    if not os.path.exists(root):
        return []
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def parse_q_num_from_filename(filename):
    m = re.search(r"q(\d)", filename.lower())
    return int(m.group(1)) if m else 1

def uncalibrated_path_from_calibrated(calibrated_fp):
    try:
        group_cal_dir = os.path.basename(os.path.dirname(os.path.dirname(calibrated_fp)))
        subject = os.path.basename(os.path.dirname(calibrated_fp))
        group = group_cal_dir.replace("_calibrated", "")
        filename = os.path.basename(calibrated_fp).replace("_preprocessed_calibrated.csv", "_preprocessed.csv")
        return data_dir(f"{group}_processed", subject, filename)
    except Exception:
        return None

def ensure_subject_output_dir(group_type, subject_name):
    base = visualization_root()
    out = os.path.join(base, f"{group_type}_calibrated", subject_name)
    os.makedirs(out, exist_ok=True)
    return out

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

def visualize_subject_all_questions_calibrated_to_file(group_type, folder):
    analyzer = import_event_analyzer()
    q_files = list_subject_q_files_calibrated(folder)
    qs = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(qs), 2, figsize=(12, 18), constrained_layout=True)

    # draw_roi_on_axes already defined above

    # draw_trajectory_on_axes already defined above

    for i, q in enumerate(qs):
        fp_cal = q_files.get(q)
        if not fp_cal:
            continue
        fp_uncal = uncalibrated_path_from_calibrated(fp_cal)
        df_cal = pd.read_csv(fp_cal) if os.path.exists(fp_cal) else None
        df_uncal = pd.read_csv(fp_uncal) if fp_uncal and os.path.exists(fp_uncal) else None
        img = load_background_image(q)
        if img is None:
            continue
        w, h = img.size
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
        # Uncalibrated (left)
        ax_l = axes[i][0]
        ax_l.imshow(img)
        draw_roi_on_axes(ax_l, w, h, kw, inst, bg)
        if df_uncal is not None and "x" in df_uncal.columns and "y" in df_uncal.columns and len(df_uncal) > 0:
            draw_trajectory_on_axes(ax_l, w, h, df_uncal)
        ax_l.set_xlim(0, w)
        ax_l.set_ylim(h, 0)
        ax_l.set_title(f"Uncalibrated Q{q}")
        ax_l.axis("off")
        # Calibrated (right)
        ax_r = axes[i][1]
        ax_r.imshow(img)
        draw_roi_on_axes(ax_r, w, h, kw, inst, bg)
        if df_cal is not None and "x" in df_cal.columns and "y" in df_cal.columns and len(df_cal) > 0:
            draw_trajectory_on_axes(ax_r, w, h, df_cal)
        ax_r.set_xlim(0, w)
        ax_r.set_ylim(h, 0)
        ax_r.set_title(f"Calibrated Q{q}")
        ax_r.axis("off")
    subject_name = os.path.basename(folder)
    fig.suptitle(f"{group_type} {subject_name}: Uncalibrated vs Calibrated", fontsize=14)
    out_dir = ensure_subject_output_dir(group_type, subject_name)
    out_path = os.path.join(out_dir, f"comparison_{subject_name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def visualize_groups(groups):
    saved = []
    for group_type in groups:
        subjects = list_subject_folders_calibrated(group_type)
        if not subjects:
            print(f"Warning: No calibrated subject folders found for group '{group_type}' in {calibration_dir(f'{group_type}_calibrated')}")
            continue
        print(f"Found {len(subjects)} subjects for group '{group_type}'. Processing...")
        for folder in subjects:
            try:
                res = visualize_subject_all_questions_calibrated_to_file(group_type, folder)
                if res:
                    saved.append(res)
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
    return saved

def main():
    results = visualize_groups(GROUP_TYPES)
    print(results)

if __name__ == "__main__":
    main()
