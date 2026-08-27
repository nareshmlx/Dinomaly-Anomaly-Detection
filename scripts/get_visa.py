#!/usr/bin/env python
"""Download VisA pcb1-4 (RGB + masks) into an Anomalib/MVTec-style layout under data/visa/.

Source: HF mirror `BrachioLab/visa` (VisA, CC BY 4.0). Regenerates the data the repo does NOT ship.
Usage:  python scripts/get_visa.py                 # all four PCB categories
        python scripts/get_visa.py --categories pcb3
Layout produced:
    data/visa/<pcbN>/train/good/*.png
    data/visa/<pcbN>/test/good/*.png
    data/visa/<pcbN>/test/bad/*.png
    data/visa/<pcbN>/ground_truth/bad/*_mask.png
"""
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data" / "visa"


def label_is_good(ds, lbl):
    feat = ds.features.get("label")
    try:
        return any(k in feat.int2str(int(lbl)).lower() for k in ("good", "normal"))
    except Exception:
        return str(lbl).strip().lower() in ("0", "good", "normal", "false")


def fetch(categories):
    from datasets import load_dataset  # imported here so --help works without deps
    for c in categories:
        out_marker = BASE / c / "test" / "bad"
        if out_marker.exists() and any(out_marker.glob("*.png")):
            print(f"{c}: already present, skipping (delete data/visa/{c} to re-fetch)")
            continue
        counts = {"train/good": 0, "test/good": 0, "test/bad": 0, "masks": 0}
        for split in ("train", "test"):
            try:
                ds = load_dataset("BrachioLab/visa", split=f"{c}.{split}")
            except Exception as e:
                print(f"  {c}.{split} load fail: {e!r}")
                continue
            for i, ex in enumerate(ds):
                img, lbl, mask = ex.get("image"), ex.get("label"), ex.get("mask")
                if img is None:
                    continue
                good = split == "train" or label_is_good(ds, lbl)
                sub = "good" if good else "bad"
                d = BASE / c / split / sub
                d.mkdir(parents=True, exist_ok=True)
                fn = f"{c}_{split}_{sub}_{i:04d}.png"
                img.convert("RGB").save(d / fn)
                counts[f"{split}/{sub}"] += 1
                if split == "test" and not good and mask is not None:
                    md = BASE / c / "ground_truth" / "bad"
                    md.mkdir(parents=True, exist_ok=True)
                    try:
                        mask.convert("L").save(md / f"{c}_{split}_{sub}_{i:04d}_mask.png")
                        counts["masks"] += 1
                    except Exception:
                        pass
        print(f"{c}: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    args = ap.parse_args()
    print(f"Downloading VisA {args.categories} -> {BASE}")
    fetch(args.categories)
    print("done.")
