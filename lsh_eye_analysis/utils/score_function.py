import numpy as np

def get_dt(df):
    """
    Calculate time interval dt[i] for each row.
    """
    if "time_diff" in df.columns:
        dt = df["time_diff"].to_numpy()
        return np.where(np.isnan(dt), 0.0, dt)

    if "milliseconds" in df.columns:
        ms = df["milliseconds"].to_numpy()
        d = np.diff(ms, prepend=ms[0])
        return np.where(d < 0, 0.0, d)

    return np.ones(len(df), dtype=float)


def in_any_roi(xs, ys, roi_list):
    """
    Check if points (xs, ys) are in any ROI in roi_list.
    """
    if not roi_list:
        return np.zeros_like(xs, dtype=bool)

    inside = np.zeros_like(xs, dtype=bool)
    for _, xmn, ymn, xmx, ymy in roi_list:
        inside |= (xs >= xmn) & (xs <= xmx) & (ys >= ymn) & (ys <= ymy)
    return inside


def enter_count(mask):
    """
    Count False -> True transitions in mask.
    """
    if len(mask) < 2:
        return 0
    prev = mask[:-1]
    curr = mask[1:]
    return int(np.sum((~prev) & curr))


def apply_offset(df, dx, dy):
    """
    Apply offset (dx, dy) to x, y columns in df and clip to [0, 1].
    """
    out = df.copy()
    if "x" not in out.columns or "y" not in out.columns:
        return out

    xs = out["x"].to_numpy(dtype=float)
    ys = out["y"].to_numpy(dtype=float)

    xs = np.clip(xs + dx, 0.0, 1.0)
    ys = np.clip(ys + dy, 0.0, 1.0)

    out["x"] = xs
    out["y"] = ys
    return out


def _sigmoid_np(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def soft_rect_prob(xs, ys, roi_list, k=40.0):
    """
    Calculate soft membership probability for points (xs, ys) against a list of ROIs.
    Returns a vector of probabilities of same length as xs.
    p_total = clip(sum(p_rect), 0, 1)
    """
    if not roi_list:
        return np.zeros_like(xs, dtype=float)
    
    p_total = np.zeros_like(xs, dtype=float)
    for _, xmn, ymn, xmx, ymy in roi_list:
        px1 = _sigmoid_np(k * (xs - xmn))
        px2 = _sigmoid_np(k * (xmx - xs))
        py1 = _sigmoid_np(k * (ys - ymn))
        py2 = _sigmoid_np(k * (ymy - ys))
        p_total += px1 * px2 * py1 * py2
        
    return np.clip(p_total, 0.0, 1.0)


def calculate_score_and_metrics(df, dx, dy, roi_kw, roi_inst, weights=None, score_type="hard"):
    """
    Calculate score and metrics for a given offset (dx, dy).
    score_type: "hard" or "soft"
    """
    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return -np.inf, {}

    if weights is None:
        weights = {}

    w_inst = float(weights.get("inst_time", 1.0))
    w_kw = float(weights.get("kw_time", 1.0))
    w_bg = float(weights.get("bg_time", 0.5))
    
    xs_np = df["x"].to_numpy(dtype=float)
    ys_np = df["y"].to_numpy(dtype=float)
    dt = get_dt(df)
    
    xs = np.clip(xs_np + dx, 0.0, 1.0)
    ys = np.clip(ys_np + dy, 0.0, 1.0)
    
    if score_type == "soft":
        # Soft ROI logic
        p_kw = soft_rect_prob(xs, ys, roi_kw, k=40.0)
        p_inst = soft_rect_prob(xs, ys, roi_inst, k=40.0)
        
        p_pos = np.clip(p_kw + p_inst, 0.0, 1.0)
        p_bg = 1.0 - p_pos
        
        t_kw = float(np.sum(dt * p_kw))
        t_inst = float(np.sum(dt * p_inst))
        t_bg = float(np.sum(dt * p_bg))
        
        # Soft entry counts are harder to define exactly like hard boundaries.
        # We can threshold them or just leave them as 0 or approximate.
        # For consistency with return signature, we'll use thresholded (hard-like) enter counts
        # or just 0. Let's use threshold 0.5 for enter counts to provide some metric.
        m_kw_hard = p_kw > 0.5
        m_inst_hard = p_inst > 0.5
        m_bg_hard = p_bg > 0.5
        
        e_kw = enter_count(m_kw_hard)
        e_inst = enter_count(m_inst_hard)
        e_bg = enter_count(m_bg_hard)
        
    else:
        # Hard ROI logic
        m_kw = in_any_roi(xs, ys, roi_kw)
        m_inst = in_any_roi(xs, ys, roi_inst)
        
        m_pos = m_kw | m_inst
        m_bg = ~m_pos
        
        t_kw = float(np.sum(dt[m_kw])) if len(dt) else 0.0
        t_inst = float(np.sum(dt[m_inst])) if len(dt) else 0.0
        t_bg = float(np.sum(dt[m_bg])) if len(dt) else 0.0
        
        e_kw = enter_count(m_kw)
        e_inst = enter_count(m_inst)
        e_bg = enter_count(m_bg)
    
    num = w_inst * t_inst + w_kw * t_kw
    denom = num + w_bg * t_bg
    score = num / denom if denom > 1e-12 else 0.0
    
    metrics = {
        "inst_time": t_inst,
        "kw_time": t_kw,
        "bg_time": t_bg,
        "time_ratio": score,
        "inst_enter": e_inst,
        "kw_enter": e_kw,
        "bg_enter": e_bg,
    }
    
    return score, metrics


def calculate_score_grid(df, roi_kw, roi_inst, dx_bounds=(-0.05, 0.05), dy_bounds=(-0.05, 0.05), step=0.005, weights=None, score_type="hard"):
    """
    Calculate score for a grid of offsets.
    score_type: "hard" or "soft"
    """
    if weights is None:
        weights = {}

    w_inst = float(weights.get("inst_time", 1.0))
    w_kw = float(weights.get("kw_time", 1.0))
    w_bg = float(weights.get("bg_time", 0.5))

    if "x" not in df.columns or "y" not in df.columns or len(df) == 0:
        return {"dx": 0.0, "dy": 0.0, "score": -np.inf, "metrics": {}}

    xs0 = df["x"].to_numpy(dtype=float)
    ys0 = df["y"].to_numpy(dtype=float)
    dt = get_dt(df)

    dx_vals = np.arange(dx_bounds[0], dx_bounds[1] + 1e-12, step)
    dy_vals = np.arange(dy_bounds[0], dy_bounds[1] + 1e-12, step)

    # Broadcast: (N, Dx, Dy)
    xs_adj = np.clip(xs0[:, None, None] + dx_vals[None, :, None], 0.0, 1.0)
    ys_adj = np.clip(ys0[:, None, None] + dy_vals[None, None, :], 0.0, 1.0)
    
    # Time weights: (N, 1, 1)
    dt3 = dt[:, None, None]

    # Pre-calculate the full shape (N, Dx, Dy) for correct mask initialization
    full_shape = (len(xs0), len(dx_vals), len(dy_vals))
    
    if score_type == "soft":
        # Soft Grid Search
        k = 40.0
        def compute_soft_mask(roi_list):
            if not roi_list:
                return np.zeros(full_shape, dtype=float)
            p_total = np.zeros(full_shape, dtype=float)
            for _, xmn, ymn, xmx, ymy in roi_list:
                # xmn, xmx, etc. are scalars, xs_adj is (N, Dx, Dy)
                # So simple subtraction works fine with broadcasting
                px1 = _sigmoid_np(k * (xs_adj - xmn))
                px2 = _sigmoid_np(k * (xmx - xs_adj))
                py1 = _sigmoid_np(k * (ys_adj - ymn))
                py2 = _sigmoid_np(k * (ymy - ys_adj))
                p_total += px1 * px2 * py1 * py2
            return np.clip(p_total, 0.0, 1.0)

        p_kw = compute_soft_mask(roi_kw)
        p_inst = compute_soft_mask(roi_inst)
        p_pos = np.clip(p_kw + p_inst, 0.0, 1.0)
        p_bg = 1.0 - p_pos
        
        t_kw = (dt3 * p_kw).sum(axis=0)
        t_inst = (dt3 * p_inst).sum(axis=0)
        t_bg = (dt3 * p_bg).sum(axis=0)
        
    else:
        # Hard Grid Search
        def build_mask(roi_list):
            if not roi_list:
                return np.zeros(full_shape, dtype=bool)
            m = np.zeros(full_shape, dtype=bool)
            for _, xmn, ymn, xmx, ymy in roi_list:
                m |= (xs_adj >= xmn) & (xs_adj <= xmx) & (ys_adj >= ymn) & (ys_adj <= ymy)
            return m

        m_kw = build_mask(roi_kw)
        m_inst = build_mask(roi_inst)
        m_bg = ~(m_kw | m_inst)

        t_kw = (dt3 * m_kw).sum(axis=0)
        t_inst = (dt3 * m_inst).sum(axis=0)
        t_bg = (dt3 * m_bg).sum(axis=0)

    num = w_inst * t_inst + w_kw * t_kw
    denom = num + w_bg * t_bg
    score_mat = np.where(denom > 1e-12, num / denom, 0.0)

    idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
    best_dx = float(dx_vals[idx[0]])
    best_dy = float(dy_vals[idx[1]])
    
    # Recalculate full metrics for the best offset using the same score_type
    _, metrics = calculate_score_and_metrics(df, best_dx, best_dy, roi_kw, roi_inst, weights, score_type=score_type)
    
    return {
        "dx": best_dx,
        "dy": best_dy,
        "score": metrics["time_ratio"],
        "metrics": metrics
    }