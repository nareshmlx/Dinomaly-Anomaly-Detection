"""Runtime for PCB anomaly inspection, fitted on good boards only.

    from pcb_anomaly import load, inspect
    load()
    result = inspect(cv2.imread("board.png"), board="pcb3")

Nothing here imports training or measurement code, so the inference path stands on its own, and
nothing here reads the dataset: `load()` needs only the two files in artifacts/.
"""
from .board import board_mask
from .peaks import topk_nms
from .runtime import CROP_FRAC, STATE, inspect, load

__all__ = ["load", "inspect", "STATE", "CROP_FRAC", "board_mask", "topk_nms"]
