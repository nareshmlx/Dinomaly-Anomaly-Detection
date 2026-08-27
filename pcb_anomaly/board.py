"""Finding the board in the frame.

Plain OpenCV colour segmentation. No model, no learned parameters.

This is the most dataset-specific code here. It works because all four VisA PCBs are blue and the two
backgrounds in that dataset are grey card and green felt, so one hue window separates board from
not-board. A green PCB, a black PCB or a bare metal part breaks it immediately. The proper fix on a
real line is not a better hue range, it is a fixture: with a known background and a fixed part
position the region of interest becomes a constant and this file disappears.
"""
import cv2
import numpy as np

BLUE_LO = (85, 60, 40)
BLUE_HI = (140, 255, 255)


def board_mask(img):
    """Boolean mask of the PCB itself, or None if the segmentation looks implausible.

    Returning None matters. Without a mask the anomaly map's argmax can land on the background, which
    is never a useful inspection target, but a confidently wrong mask is worse than none at all. So the
    caller treats None as "do not constrain the peak" and carries on.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    if not 0.05 < filled.mean() / 255.0 < 0.85:
        return None
    # Dilate before returning. Components overhang the blue substrate: LED domes, connector shells,
    # the ultrasonic transducers on pcb1. Without this, a bent pin sitting just off the blue would be
    # masked out and the peak could never land on it.
    return cv2.dilate(filled, np.ones((41, 41), np.uint8)) > 0
