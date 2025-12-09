import os
import argparse

def find_files(root, name, ext):
    results = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.startswith(name) and (not ext or f.lower().endswith('.' + ext.lower())):
                results.append(os.path.join(dirpath, f))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./lsh_eye_analysis/data/data_calibration')
    parser.add_argument('--name', default='5_preprocessed_calibrated')
    parser.add_argument('--ext', default='csv')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    root = args.root
    files = find_files(root, args.name, args.ext)
    print(f'Found {len(files)} files to delete')
    for p in files:
        if args.dry_run:
            print(f'[DRY-RUN] {p}')
        else:
            try:
                os.remove(p)
                print(f'Deleted {p}')
            except Exception as e:
                print(f'Failed {p}: {e}')

if __name__ == '__main__':
    main()

