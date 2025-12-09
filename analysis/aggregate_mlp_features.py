import os
import re
import pandas as pd
import numpy as np

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def parse_q_num(name):
    m = re.search(r"q(\d)", name.lower())
    return int(m.group(1)) if m else 1

def roi_cat(s):
    if not isinstance(s, str):
        return "NONE"
    s = s.upper()
    if s.startswith("KW_"):
        return "KW"
    if s.startswith("INST_"):
        return "INST"
    if s.startswith("BG_"):
        return "BG"
    return "OTHER"

def aggregate_subject(df):
    subj = df["subject"].iloc[0] if len(df) else ""
    grp = df["group"].iloc[0] if len(df) else ""
    q = df["ADQ_ID"].iloc[0] if len(df) else ""
    qn = parse_q_num(str(q))
    df["ROI_CAT"] = [roi_cat(x) for x in df["ROI"]]
    fx = df[df["EventType"] == "fixation"].copy()
    sc = df[df["EventType"] == "saccade"].copy()
    fx_dur = float(fx["Duration_ms"].sum()) if len(fx) else 0.0
    sc_dur = float(sc["Duration_ms"].sum()) if len(sc) else 0.0
    fx_cnt = int(len(fx))
    sc_cnt = int(len(sc))
    fx_amp_mean = float(fx["Amplitude_deg"].mean()) if len(fx) else np.nan
    sc_amp_mean = float(sc["Amplitude_deg"].mean()) if len(sc) else np.nan
    sc_maxvel_mean = float(sc["MaxVel"].mean()) if len(sc) else np.nan
    sc_meanvel_mean = float(sc["MeanVel"].mean()) if len(sc) else np.nan
    kw_fx_dur = float(fx[fx["ROI_CAT"] == "KW"]["Duration_ms"].sum()) if len(fx) else 0.0
    inst_fx_dur = float(fx[fx["ROI_CAT"] == "INST"]["Duration_ms"].sum()) if len(fx) else 0.0
    bg_fx_dur = float(fx[fx["ROI_CAT"] == "BG"]["Duration_ms"].sum()) if len(fx) else 0.0
    total_dur = fx_dur + sc_dur
    kw_ratio = (kw_fx_dur / total_dur) if total_dur > 0 else np.nan
    inst_ratio = (inst_fx_dur / total_dur) if total_dur > 0 else np.nan
    bg_ratio = (bg_fx_dur / total_dur) if total_dur > 0 else np.nan
    out = {
        "group": grp,
        "subject": subj,
        "q": qn,
        "fixation_count": fx_cnt,
        "saccade_count": sc_cnt,
        "fixation_duration_ms": fx_dur,
        "saccade_duration_ms": sc_dur,
        "fixation_amplitude_mean_deg": fx_amp_mean,
        "saccade_amplitude_mean_deg": sc_amp_mean,
        "saccade_maxvel_mean": sc_maxvel_mean,
        "saccade_meanvel_mean": sc_meanvel_mean,
        "kw_fix_duration_ms": kw_fx_dur,
        "inst_fix_duration_ms": inst_fx_dur,
        "bg_fix_duration_ms": bg_fx_dur,
        "kw_time_ratio": kw_ratio,
        "inst_time_ratio": inst_ratio,
        "bg_time_ratio": bg_ratio,
    }
    return out

def aggregate_file(fp):
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp)
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("subject")
    rows = [aggregate_subject(gdf.copy()) for _, gdf in g]
    return pd.DataFrame(rows)

def aggregate_for_group(source_root, group):
    out_root = ensure_dir(os.path.join(project_root(), "data", "MLP_data", "features", "event_features", f"{group}_group"))
    saved = {}
    for q in [1, 2, 3, 4, 5]:
        fp = os.path.join(source_root, f"{group}_group", f"{group}_q{q}.csv")
        df_out = aggregate_file(fp)
        out_fp = os.path.join(out_root, f"{group}_q{q}.csv")
        df_out.to_csv(out_fp, index=False)
        saved[q] = out_fp
    return saved

def aggregate_all(groups):
    source_root = os.path.join(project_root(), "data", "MLP_data", "event_data")
    res = {}
    for g in groups:
        res[g] = aggregate_for_group(source_root, g)
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=str, default="control,mci,ad")
    args = parser.parse_args()
    groups = [s.strip() for s in args.groups.split(",") if s.strip()]
    result = aggregate_all(groups)
    print(result)

