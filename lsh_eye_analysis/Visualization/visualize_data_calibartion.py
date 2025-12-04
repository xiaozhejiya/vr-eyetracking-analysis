"""
可视化 Q1–Q5 背景图的 ROI，并叠加已校准的眼动轨迹
- 通过常量 `GROUP_TYPE` 指定组别，从 `lsh_eye_analysis/data_calibartion/{group}_calibrated/{subject_folder}` 读取数据
- ROI 定义来自 `analysis.event_analyzer.EventAnalyzer`
- 输出文件保存到当前目录下的 `outputs`
"""
import argparse
import os
import re
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

GROUP_TYPE = "mci"

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

def draw_roi_layer(base_img, kw, inst, bg, roi_type="hard"):
    """
    Draw ROI layer on the base image.
    roi_type: "hard" or "soft"
    """
    if roi_type == "soft":
        return draw_soft_roi_layer(base_img, kw, inst, bg, k=40.0)

    w, h = base_img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    
    d = ImageDraw.Draw(layer)
    try:
        font_ = ImageFont.truetype("arial.ttf", 14)
    except:
        font_ = ImageFont.load_default()

    def draw_rois(roi_list, color, alpha):
        if not roi_list: return
        for rn, xmn, ymn, xmx, ymy in roi_list:
            x1 = int(xmn * w)
            x2 = int(xmx * w)
            y_top = int((1 - ymn) * h)
            y_bot = int((1 - ymy) * h)
            y1, y2 = min(y_top, y_bot), max(y_top, y_bot)
            
            # Draw rectangle
            d.rectangle([(x1, y1), (x2, y2)], fill=(color[0], color[1], color[2], alpha), outline=(color[0], color[1], color[2], 255), width=2)
            
            # Draw label
            try:
                # Pillow >= 10.0.0
                tw = d.textlength(rn, font=font_)
                th = font_.size 
            except AttributeError:
                # Pillow < 10.0.0
                tw, th = d.textsize(rn, font=font_)
                
            tx = min(x1 + 2, w - tw)
            ty = min(y2 + 2, h - th)
            d.rectangle([(tx, ty), (tx + tw, ty + th)], fill=(255, 255, 255, 160))
            d.text((tx, ty), rn, fill=(0, 0, 0, 255), font=font_)

    draw_rois(bg, (0, 128, 255), 60)
    draw_rois(inst, (255, 165, 0), 80)
    draw_rois(kw, (255, 0, 0), 100)
    
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

def visualize_five_backgrounds(roi_type="hard"):
    analyzer = import_event_analyzer()
    out_dir = ensure_output_dir()
    results = []
    for q in [1, 2, 3, 4, 5]:
        img = load_background_image(q)
        if img is None:
            continue
        
        # Determine which ROI definition to fetch
        # Assuming get_roi_def returns the standard rectangular ROIs.
        # If soft ROIs are fundamentally different (e.g. probabilistic maps),
        # the EventAnalyzer needs to support fetching them.
        # Based on typical implementations in this project, "soft" often refers to 
        # the scoring method (sigmoid boundaries) rather than distinct rect definitions.
        # However, if there's a "soft" parameter in get_roi_def, we should use it.
        # Checking EventAnalyzer signature would be ideal, but assuming standard behavior:
        kw, inst, bg = analyzer.get_roi_def(f"n2q{q}")
        
        roi_layer = draw_roi_layer(img, kw, inst, bg, roi_type=roi_type)
        combined = Image.alpha_composite(img.convert("RGBA"), roi_layer).convert("RGB")
        out_path = os.path.join(out_dir, f"background_Q{q}_{roi_type}_roi.jpg")
        combined.save(out_path)
        results.append(out_path)
    return results

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

def visualize_subject_all_questions_calibrated(group_type, roi_type="hard"):
    folder = pick_random_subject_folder_calibrated(group_type, require_complete=True)
    if not folder:
        return []
    analyzer = import_event_analyzer()
    q_files = list_subject_q_files_calibrated(folder)
    qs = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(qs), 2, figsize=(12, 18), constrained_layout=True)

    def draw_roi_on_axes(ax, w, h, roi_kw, roi_inst, roi_bg):
        if roi_type == "soft":
            # Use matplotlib imshow for soft maps
            map_inst = build_soft_map_for_roi_list(w, h, roi_inst, k=40.0)
            map_kw   = build_soft_map_for_roi_list(w, h, roi_kw,   k=40.0)
            map_pos = np.clip(map_inst + map_kw, 0.0, 1.0)
            map_bg  = 1.0 - map_pos
            
            # Create RGBA image for overlay
            rgba = np.zeros((h, w, 4))
            
            # Add colors (using same colors as draw_soft_roi_layer but normalized to 0-1)
            # BG: Blue (0, 128, 255) -> (0, 0.5, 1.0)
            # Inst: Orange (255, 165, 0) -> (1.0, 0.65, 0)
            # KW: Red (255, 0, 0) -> (1.0, 0, 0)
            
            # We need to blend these layers. A simple additive approach for visualization:
            # (This is a simplified blending compared to PIL alpha_composite)
            
            bg_color = np.array([0, 0.5, 1.0])
            inst_color = np.array([1.0, 0.65, 0])
            kw_color = np.array([1.0, 0.0, 0.0])
            
            # Base alpha
            bg_alpha = 0.35  # 90/255
            inst_alpha = 0.63 # 160/255
            kw_alpha = 0.78   # 200/255
            
            # Accumulate colors weighted by their map values and alphas
            # Note: This is an approximation for matplotlib display
            
            # Initialize with transparent
            final_img = np.zeros((h, w, 4))
            
            # Helper to blend layer on top
            def blend_layer(base, color, map_val, alpha_max):
                # alpha for this pixel
                alpha = map_val[..., None] * alpha_max
                # color for this pixel
                rgb = color.reshape(1,1,3)
                
                # Standard alpha blending: out = src * alpha + dst * (1 - alpha)
                src_rgb = rgb
                dst_rgb = base[..., :3]
                dst_a = base[..., 3:]
                
                out_a = alpha + dst_a * (1 - alpha)
                # Avoid division by zero
                mask = out_a > 0
                out_rgb = np.zeros_like(dst_rgb)
                
                # Only calculate where alpha > 0
                # out_rgb = (src_rgb * alpha + dst_rgb * dst_a * (1 - alpha)) / out_a
                
                # Simplified: just overlay "softly" by summing for visual check
                # Since we want to show the heatmaps.
                # Let's use imshow extent with alpha maps instead of manual blending for simplicity?
                # No, manual blending gives better control over "heatmap" look.
                
                pass

            # Let's use the PIL function we already wrote to generate the overlay image,
            # then convert to numpy for matplotlib. This ensures consistency.
            layer_pil = draw_soft_roi_layer(Image.new("RGB", (w, h)), roi_kw, roi_inst, roi_bg, k=40.0)
            layer_np = np.array(layer_pil) / 255.0
            
            ax.imshow(layer_np, extent=[0, w, h, 0]) # Note: y-origin is top for images, but we might need check
            # draw_soft_roi_layer returns an image where (0,0) is top-left.
            # matplotlib imshow default origin is upper.
            # But our draw_rois (hard) uses Rectangle with y flipped (1-ymn)*h.
            # Let's check draw_soft_roi_layer logic:
            # ys_img = (np.arange(h) + 0.5) / float(h) -> 0..1 top to bottom
            # ys = 1.0 - ys_img -> 1..0 (bottom is 0 in norm, top is 1 in norm)
            # ROI ymn is 0 at bottom.
            # So soft logic matches standard mathematical coordinates (0 bottom), mapped to image (0 top).
            # So displaying the resulting image with imshow(origin='upper') is correct.
            
            return

        def draw_rois(roi_list, color, alpha):
            if not roi_list: return
            for rn, xmn, ymn, xmx, ymy in roi_list:
                x1 = xmn * w
                x2 = xmx * w
                y_top = (1 - ymn) * h
                y_bot = (1 - ymy) * h
                y = min(y_top, y_bot)
                height = abs(y_top - y_bot)
                rect = Rectangle((x1, y), x2 - x1, height, linewidth=2,
                                 edgecolor=(color[0] / 255, color[1] / 255, color[2] / 255, 1.0),
                                 facecolor=(color[0] / 255, color[1] / 255, color[2] / 255, alpha / 255))
                ax.add_patch(rect)
        
        # Hard ROI: standard colors
        draw_rois(roi_bg, (0, 128, 255), 60)
        draw_rois(roi_inst, (255, 165, 0), 80)
        draw_rois(roi_kw, (255, 0, 0), 100)

    def draw_trajectory_on_axes(ax, w, h, df, point_size=8, line_width=2):
        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()
        xp = xs * w
        yp = (1 - ys) * h
        ax.plot(xp, yp, color=(50 / 255, 200 / 255, 120 / 255, 0.7), linewidth=line_width)
        ax.scatter(xp, yp, s=point_size, color=(0, 0, 1, 0.6))
        if len(xp):
            ax.scatter([xp[0]], [yp[0]], s=30, color=(0, 1, 0, 0.9))
            ax.scatter([xp[-1]], [yp[-1]], s=30, color=(1, 0, 0, 0.9))

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
        ax_l.set_title(f"Uncalibrated Q{q} ({roi_type})")
        ax_l.axis("off")
        
        # Calibrated (right)
        ax_r = axes[i][1]
        ax_r.imshow(img)
        draw_roi_on_axes(ax_r, w, h, kw, inst, bg)
        if df_cal is not None and "x" in df_cal.columns and "y" in df_cal.columns and len(df_cal) > 0:
            draw_trajectory_on_axes(ax_r, w, h, df_cal)
        ax_r.set_xlim(0, w)
        ax_r.set_ylim(h, 0)
        ax_r.set_title(f"Calibrated Q{q} ({roi_type})")
        ax_r.axis("off")
        
    fig.suptitle(f"{group_type} subject: Uncalibrated vs Calibrated ({roi_type} ROI)", fontsize=14)
    return []

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
    parser = argparse.ArgumentParser(description="Visualize Calibration with ROI options")
    # 使用当前文件顶部的 GROUP_TYPE 作为默认值
    parser.add_argument("--roi-type", type=str, choices=["hard", "soft"], default="soft", help="ROI visualization type (hard/soft)")
    parser.add_argument("--group", type=str, default=GROUP_TYPE, help="Group type (e.g. mci, control)")
    args = parser.parse_args()

    # 直接使用 args 里的值，不修改全局变量，避免语法错误
    current_group_type = args.group
    current_roi_type = args.roi_type

    print(f"Visualizing for Group: {current_group_type}, ROI Type: {current_roi_type}")
    visualize_subject_all_questions_calibrated(current_group_type, roi_type=current_roi_type)
    plt.show()

if __name__ == "__main__":
    main()