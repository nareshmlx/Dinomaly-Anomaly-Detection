#!/usr/bin/env python
"""Reproduce the accept/reject numbers on the VisA test split.

    python scripts/measure_metrics.py              # accuracy, per product and pooled
    python scripts/measure_metrics.py --latency    # timing distribution instead

This exists so the figures in docs/results.md are checkable rather than claimed. It runs the same
`pcb_anomaly.inspect` the notebooks call over every test image of every VisA product you have
downloaded, and writes results/endtoend.json.

The product name comes from the directory the image sits in and is handed to `inspect`, because the
accept/reject limit is set per product. Nothing here recognises the part from the image.

Across all four published products it reports 97.3 percent of defective boards rejected, 11 missed
out of 400, and 94.0 percent of good boards accepted, 24 false alarms out of 402. That takes about a
minute on an RTX 5090 once the data is on disk.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb_anomaly import inspect, load                                          # noqa: E402

DATA = ROOT / "data/visa"
OUT = ROOT / "results"
ALL_PARTS = ["pcb1", "pcb2", "pcb3", "pcb4"]


def parts_on_disk():
    """Whichever products were downloaded. Both notebooks fetch only pcb3, so four is not a given."""
    found = [p for p in ALL_PARTS if any((DATA / p / "test" / "bad").glob("*.png"))]
    if not found:
        sys.exit(f"no images under {DATA}. Run: python scripts/get_visa.py")
    return found


def wilson(k, n, z=1.96):
    """Wilson score interval. Each product contributes 100 defective boards, and a bare percentage
    from a sample that size hides how wide the uncertainty really is."""
    if n == 0:
        return [float("nan")] * 2
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def pct(v, q):
    return round(float(np.percentile(v, q)), 1)


def block(rows):
    bad = [r for r in rows if r["truth"] == "bad"]
    good = [r for r in rows if r["truth"] == "good"]
    tp = sum(r["verdict"] == "FAIL" for r in bad)
    tn = sum(r["verdict"] == "PASS" for r in good)
    return {
        "n_bad": len(bad), "n_good": len(good),
        "sensitivity": tp / len(bad) if bad else None,
        "sensitivity_ci": wilson(tp, len(bad)) if bad else None,
        "missed": len(bad) - tp,
        "specificity": tn / len(good) if good else None,
        "specificity_ci": wilson(tn, len(good)) if good else None,
        "false_alarms": len(good) - tn,
        "median_ms": pct([r["ms"] for r in rows], 50),
    }


def collect(files):
    rows, errors = [], 0
    inspect(cv2.imread(str(files[0][2])), files[0][0])   # discarded, it pays for kernel warm-up
    t0 = time.time()
    for i, (part, split, path) in enumerate(files, 1):
        try:
            r = inspect(cv2.imread(str(path)), part)
        except Exception as exc:
            errors += 1
            print(f"  error on {path.name}: {exc}")
            continue
        rows.append({"part": part, "truth": split, "score": r["score"],
                     "verdict": r["verdict"], "ms": r["ms_total"]})
        if i % 200 == 0:
            print(f"  [{time.time() - t0:4.0f}s] {i}/{len(files)}", flush=True)
    return rows, errors


def end_to_end(parts):
    files = [(p, s, f)
             for p in parts for s in ("good", "bad")
             for f in sorted((DATA / p / "test" / s).glob("*.png"))]

    rows, errors = collect(files)
    result = {
        "n": len(rows), "errors": errors, "parts": parts,
        "note": "VisA test split through pcb_anomaly.inspect, product name supplied per image",
        "ALL": block(rows),
        "per_part": {p: block([r for r in rows if r["part"] == p]) for p in parts},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "endtoend.json").write_text(json.dumps({**result, "records": rows}, indent=1))

    a = result["ALL"]
    print(f"\n{len(rows)} images across {', '.join(parts)}, {errors} errors")
    print(f"  defective boards rejected  {a['sensitivity']:.4f}  "
          f"[{a['sensitivity_ci'][0]:.3f}, {a['sensitivity_ci'][1]:.3f}]   missed {a['missed']}/{a['n_bad']}")
    print(f"  good boards accepted       {a['specificity']:.4f}  "
          f"[{a['specificity_ci'][0]:.3f}, {a['specificity_ci'][1]:.3f}]   "
          f"false alarms {a['false_alarms']}/{a['n_good']}")
    print(f"  median latency             {a['median_ms']:.0f} ms")
    print("\n  per product:")
    for part, v in result["per_part"].items():
        print(f"    {part}  rejects {v['sensitivity']:.3f} ({v['missed']} missed)   "
              f"accepts {v['specificity']:.3f} ({v['false_alarms']} false alarms)")
    print(f"\n  wrote {(OUT / 'endtoend.json').relative_to(ROOT)}")


def latency(parts, n_per_part):
    rng = random.Random(1)
    files = []
    for p in parts:
        for s in ("good", "bad"):
            candidates = sorted((DATA / p / "test" / s).glob("*.png"))
            files += [(p, s, f) for f in rng.sample(candidates, min(n_per_part, len(candidates)))]
    rng.shuffle(files)

    rows, _ = collect(files)
    total = [r["ms"] for r in rows]
    result = {
        "n": len(rows), "parts": parts,
        "note": ("ms_total spans the forward pass, the board segmentation and peak finding. Batch of "
                 "one, no optimisation, warm-up call discarded. Rendering is not included, since this "
                 "path does not draw pictures."),
        "total_ms": {"median": pct(total, 50), "mean": round(float(np.mean(total)), 1),
                     "p90": pct(total, 90), "p99": pct(total, 99),
                     "min": int(min(total)), "max": int(max(total))},
        "per_part_median_ms": {p: pct([r["ms"] for r in rows if r["part"] == p], 50) for p in parts},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(result, indent=1))
    print("\n" + json.dumps(result, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", action="store_true", help="timing distribution instead of accuracy")
    ap.add_argument("--n-per-part", type=int, default=25, help="latency only, per product per class")
    args = ap.parse_args()
    parts = parts_on_disk()
    load()
    if args.latency:
        latency(parts, args.n_per_part)
    else:
        end_to_end(parts)
