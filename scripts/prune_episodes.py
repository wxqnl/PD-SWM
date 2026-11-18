"""Delete the oldest saved training episodes of long runs, keeping the newest --keep_steps steps per run.

dreamer.py replays only the newest dataset_size steps (1M) and, on a restart, loads only those from train_eps;
older episode files are never read again. dreamer.py counts its env steps from the saved episodes, so the
steps of the deleted ones go to train_eps/pruned_steps.txt, which count_steps adds. Minecraft writes about
3.5 GB of episodes per 1M steps, so 8 runs of 15M+ steps would fill the disk. Files are named
<time>-<id>-<length>.npz; the name order is the time order that tools.load_episodes uses. Stdlib only.

usage: python3 scripts/prune_episodes.py <run dir> ... [--keep_steps 1500000] [--every 1800]
"""
import argparse
import pathlib
import time


def prune(run, keep_steps):
    folder = pathlib.Path(run, "train_eps")
    files = sorted(folder.glob("*.npz"), reverse=True)
    kept = 0
    drop = []
    for f in files:
        if kept < keep_steps:
            kept += int(f.stem.rsplit("-", 1)[1]) - 1
        else:
            drop.append(f)
    if not drop:
        return kept, 0, 0
    # dreamer.py counts its env steps from the saved episodes: record the deleted ones first (count_steps adds them)
    record = folder / "pruned_steps.txt"
    total = (int(record.read_text()) if record.exists() else 0) + sum(int(f.stem.rsplit("-", 1)[1]) - 1 for f in drop)
    tmp = folder / "pruned_steps.txt.tmp"
    tmp.write_text(str(total))
    tmp.replace(record)
    freed = sum(f.stat().st_size for f in drop)
    for f in drop:
        f.unlink()
    return kept, len(drop), freed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--keep_steps", type=int, default=1_500_000)
    parser.add_argument("--every", type=int, default=0, help="seconds between passes; 0 = one pass")
    args = parser.parse_args()
    assert args.keep_steps > 1_000_000, "keep more than the replay (dataset_size 1M)"
    while True:
        for run in args.runs:
            kept, removed, freed = prune(run, args.keep_steps)
            if removed:
                print(f"{time.strftime('%F %T')} {run}: removed {removed} episodes ({freed / 1e9:.1f} GB), "
                      f"kept {kept} steps", flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
