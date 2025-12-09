import os
import sys
import shutil
import os

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.append(project_root())

from data_processing.vr_eyetracking_processor import process_directory

if __name__ == "__main__":
    if os.environ.get('SYNC_LEGACY_PROCESSED') == '1':
        legacy_roots = {
            'mci': r"./data/mci_processed",
            'control': r"./data/control_processed",
            'ad': r"./data/ad_processed",
        }
        target_roots = {
            'mci': r"./lsh_eye_analysis/data/data_processed/mci_processed",
            'control': r"./lsh_eye_analysis/data/data_processed/control_processed",
            'ad': r"./lsh_eye_analysis/data/data_processed/ad_processed",
        }

        for group, src in legacy_roots.items():
            if os.path.exists(src):
                dst = target_roots[group]
                for sub in os.listdir(src):
                    sub_src = os.path.join(src, sub)
                    if os.path.isdir(sub_src):
                        sub_dst = os.path.join(dst, sub)
                        os.makedirs(sub_dst, exist_ok=True)
                        for f in os.listdir(sub_src):
                            if f.endswith('.csv'):
                                shutil.copy2(os.path.join(sub_src, f), os.path.join(sub_dst, f))
                print(f"synced legacy {group}_processed -> {dst}")
        print("done syncing legacy processed datasets.")
        sys.exit(0)

    RAW_ROOT = r"./lsh_eye_analysis/data/data_raw"
    OUT_ROOT = r"./lsh_eye_analysis/data/data_processed"

    os.makedirs(OUT_ROOT, exist_ok=True)

    for top_group in os.listdir(RAW_ROOT):
        top_in = os.path.join(RAW_ROOT, top_group)
        if not os.path.isdir(top_in):
            continue
        for sub in os.listdir(top_in):
            sub_in = os.path.join(top_in, sub)
            if os.path.isdir(sub_in):
                sub_out = os.path.join(OUT_ROOT, top_group.replace('_raw', '_processed'), sub)
                gt = top_group.replace('_raw', '')
                if gt == 'control':
                    token = 'n'
                elif gt == 'mci':
                    token = 'm'
                else:
                    token = gt
                num = None
                try:
                    num = int(''.join([c for c in sub if c.isdigit()]))
                except Exception:
                    num = None
                prefix = f"{token}{num}q" if num is not None else ""
                stats = process_directory(sub_in, sub_out, file_prefix=prefix, file_suffix="_preprocessed")
                print(f"{top_group}/{sub}: {stats}")
    
