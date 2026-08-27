#!/usr/bin/env python
"""Merge the four PCB products into one dataset so they train as a single model.

    python scripts/build_all4.py

Dinomaly is a multi-class method: one checkpoint covers several products rather than
one checkpoint per product. anomalib's Folder datamodule reads a single directory, so
the four products have to look like one. This builds that view with symlinks, which
costs no disk and leaves the per-product directories untouched.

Expected result, and what the published numbers were trained on:

    data/visa/_all4/
    ├── train/good/        3,614 links   (904 + 901 + 905 + 904)
    ├── test/good/           402 links
    ├── test/bad/            400 links
    └── ground_truth/bad/    400 links

Filenames already carry their product prefix (pcb1_train_good_0000.png), so nothing
collides in the merged directory and every link keeps the name of its target.
"""
import argparse
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOARDS = ["pcb1", "pcb2", "pcb3", "pcb4"]
SPLITS = ["train/good", "test/good", "test/bad", "ground_truth/bad"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=REPO / "data/visa", type=Path,
                    help="directory holding pcb1..pcb4 (default: data/visa)")
    ap.add_argument("--name", default="_all4", help="name of the merged directory")
    ap.add_argument("--force", action="store_true", help="replace an existing merged directory")
    args = ap.parse_args()

    dest = args.data / args.name
    if dest.exists():
        if not args.force:
            raise SystemExit(f"{dest} already exists. Pass --force to rebuild it.")
        for f in sorted(dest.rglob("*"), reverse=True):
            f.unlink() if f.is_symlink() or f.is_file() else f.rmdir()
        dest.rmdir()

    missing = [b for b in BOARDS if not (args.data / b / "train/good").is_dir()]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} under {args.data}\n"
            f"  run: python scripts/get_visa.py --categories {' '.join(missing)}")

    counts = Counter()
    for split in SPLITS:
        (dest / split).mkdir(parents=True)
        for board in BOARDS:
            src = args.data / board / split
            if not src.is_dir():
                continue
            for f in sorted(src.glob("*.png")):
                # Relative links, so the tree survives the repo being moved or cloned
                # to a different path.
                rel = os.path.relpath(f.resolve(), (dest / split).resolve())
                (dest / split / f.name).symlink_to(rel)
                counts[split] += 1

    print(f"built {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
    for split in SPLITS:
        print(f"  {split:20s} {counts[split]:5d}")

    expected = {"train/good": 3614, "test/good": 402, "test/bad": 400, "ground_truth/bad": 400}
    if dict(counts) != expected:
        print("\nwarning: counts differ from the published run:")
        for k, v in expected.items():
            if counts[k] != v:
                print(f"  {k:20s} got {counts[k]}, published run used {v}")


if __name__ == "__main__":
    main()
