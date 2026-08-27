"""Turning an inspection result into pictures for a person.

The inspection logic returns numbers; this file decides how they are drawn. It borrows one constant
from runtime.py, the detector's input geometry, because the heatmap has to be placed in the same
frame as the boxes it sits next to.
"""
import cv2
import numpy as np

from .runtime import MODEL_RESIZE

BOX_PRIMARY = (0, 0, 255)
BOX_ALTERNATE = (0, 190, 255)


def heatmap(img, raw_map):
    """The anomaly map painted over the board.

    Normalisation is per image, so a board with no defect still fills the whole colour range. The
    picture is for locating a defect the score has already found, not for deciding there is one.

    The map covers the detector's centre crop, not the whole frame, so it is padded back out before
    being stretched over the image. Skip that and the colour sits a few percent away from the boxes
    drawn beside it. The border the model never saw is filled by repeating the nearest edge.
    """
    h, w = img.shape[:2]
    pad = (MODEL_RESIZE - raw_map.shape[0]) // 2
    full = cv2.copyMakeBorder(raw_map, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    scaled = cv2.resize(full, (w, h))
    coloured = cv2.applyColorMap(
        cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, coloured, 0.45, 0)


def boxes(img, peaks, half, all_three=True):
    """Draw the candidate boxes. The first is solid, the alternates are thinner and numbered.

    Only ever call this on a reject. An accepted board must not come back with boxes on it: candidate
    boxes are the station saying "look here", and on a part that passed there is nothing to look at. An
    operator reading red rectangles on an accepted board will hesitate, or stop trusting them.
    """
    out = img.copy()
    h, w = img.shape[:2]
    for i, (cx, cy) in enumerate(peaks if all_three else peaks[:1]):
        colour = BOX_PRIMARY if i == 0 else BOX_ALTERNATE
        p0 = (max(0, cx - half), max(0, cy - half))
        p1 = (min(w, cx + half), min(h, cy + half))
        cv2.rectangle(out, p0, p1, colour, 6 if i == 0 else 3)
        if i > 0 or all_three:
            cv2.putText(out, f"#{i + 1}", (p0[0] + 8, p0[1] + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3, cv2.LINE_AA)
    return out
