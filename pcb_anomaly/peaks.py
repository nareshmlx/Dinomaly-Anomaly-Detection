"""Turning the anomaly map into candidate boxes.

Zero parameters. This is arithmetic on the 392x392 map the detector returns.
"""
import numpy as np


def topk_nms(a, k, radius):
    """The k highest peaks, each suppressing a neighbourhood so the resulting crops do not overlap.

    Take the highest value, blank a square around it, take the new highest, repeat. Without the blanking
    the second peak would be the cell next door and all three boxes would land on the same defect.

    Set `radius` to the box half-width and the boxes can touch but never overlap, which is what makes
    the alternates useful: each one points somewhere genuinely different.

    Cells masked out by the caller arrive as -inf, so the loop stops early rather than returning a peak
    the caller already excluded.
    """
    a = a.astype(np.float64).copy()
    peaks = []
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(a), a.shape)
        if not np.isfinite(a[y, x]):
            break
        peaks.append((int(y), int(x)))
        y0, y1 = max(0, y - radius), min(a.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(a.shape[1], x + radius + 1)
        a[y0:y1, x0:x1] = -np.inf
    return peaks
