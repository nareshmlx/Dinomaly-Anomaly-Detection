"""The inspection runtime: load the detector once, then answer two questions per board.

    accept or reject        the board's score against that product's limit
    where is the defect     the three highest peaks on the anomaly map, constrained to the PCB

Both answers come from a single anomaly map produced by one forward pass. The model was fitted on
good boards only and has never been shown a defect, so "where is the defect" is really "where does
this board disagree most with what a good one looks like".
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .board import board_mask
from .peaks import topk_nms

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# The box side, as a fraction of image width. The NMS radius is set to half of this so boxes can touch
# but never overlap.
CROP_FRAC = 0.22

# Anomalib's Dinomaly pre-processor resizes to 448 and then centre-crops to 392, so the anomaly map
# describes the middle 392/448 of the frame and the outer 6.25 percent of every edge is never seen by
# the model. Peaks and the board mask both have to travel through that same crop, or they sit up to
# 6 percent of the image width away from what the model was actually looking at.
MODEL_RESIZE = 448

STATE = {}


def load(quiet=False):
    """Load the detector and the per-product limits into STATE. Call once, not per image."""
    from anomalib.models import Dinomaly

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    ckpt = ARTIFACTS / "detector.pt"
    if not ckpt.exists():
        raise SystemExit(
            f"missing {ckpt.relative_to(ROOT)}\n"
            f"  run: python scripts/fetch_checkpoint.py")

    say("loading the detector ...")
    model = Dinomaly(post_processor=False)
    # strict=False on purpose: the published checkpoint holds only the weights we trained, the
    # bottleneck and decoder. The frozen DINOv2 encoder comes from timm when Dinomaly is constructed,
    # so those keys are expected to be missing here.
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)

    STATE["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE["model"] = model.to(STATE["device"]).eval()
    STATE["thr"] = json.loads((ARTIFACTS / "thresholds.json").read_text())

    say(f"  device: {STATE['device'].type}"
        + ("   (no GPU found, inference will be slower)" if STATE["device"].type == "cpu" else ""))
    say("  limits: " + "  ".join(f"{k} {v:.4f}" for k, v in STATE["thr"]["per_board_p95"].items()))
    return STATE


def inspect(img, board):
    """Score one image and locate its worst regions.

    `board` names the product, because the accept/reject limit is per product. Pass one of the keys
    in artifacts/thresholds.json; anything else scores the board but returns no verdict.

    The plain keys are JSON-safe. The two underscored ones carry the arrays render.py needs and are
    meant to be dropped before anything is written out.
    """
    t0 = time.time()
    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    with torch.no_grad():
        out = STATE["model"](torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(STATE["device"]))
    amap = out.anomaly_map
    a = (amap[0, 0] if amap.dim() == 4 else amap[0]).float().cpu().numpy()
    score = float(out.pred_score[0])
    ms_detect = round((time.time() - t0) * 1000)

    limit = STATE["thr"]["per_board_p95"].get(board)
    verdict = None if limit is None else ("FAIL" if score >= limit else "PASS")

    n = a.shape[0]
    pad = (MODEL_RESIZE - n) // 2

    # Peaks are taken on the board only. A hot pixel on the background is not a defect in the part,
    # and letting one win would put a box on the bench rather than on the unit.
    raw = a.copy()
    mask = board_mask(img)
    if mask is not None:
        big = cv2.resize(mask.astype(np.uint8), (MODEL_RESIZE, MODEL_RESIZE),
                         interpolation=cv2.INTER_NEAREST)
        a = np.where(big[pad:pad + n, pad:pad + n] > 0, a, -np.inf)

    # Radius in map pixels for a box that is CROP_FRAC of the image wide: the map spans
    # MODEL_RESIZE, not the full frame, so the conversion goes through that and not through w.
    peaks = topk_nms(a, 3, max(1, int(CROP_FRAC * MODEL_RESIZE / 2)))
    xy = [(int((px + pad) * w / MODEL_RESIZE), int((py + pad) * h / MODEL_RESIZE)) for py, px in peaks]

    return {
        "board": board,
        "score": round(score, 4), "threshold": limit, "verdict": verdict,
        "peaks": xy, "board_masked": mask is not None,
        "ms_detect": ms_detect, "ms_total": round((time.time() - t0) * 1000),
        "_raw_map": raw, "_half": int(CROP_FRAC * w / 2),
    }
