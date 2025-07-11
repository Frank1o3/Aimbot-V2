import time
import numpy as np
import cv2 as cv
from typing import Tuple, Union, Sequence, List
import scipy.sparse

MatLike = Union[np.ndarray, scipy.sparse.spmatrix, Sequence[Sequence[float]]]

def precompute_hsv(target_color: Tuple[int, int, int], tolerance: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dynamic HSV bounds based on target color and tolerance for each channel."""
    target_bgr_np = np.array([[target_color]], dtype=np.uint8)
    target_hsv = cv.cvtColor(target_bgr_np, cv.COLOR_BGR2HSV)[0][0]

    h, s, v = target_hsv
    lower = np.array([
        max(0, h - tolerance),
        max(0, s - tolerance),
        max(0, v - tolerance)
    ], dtype=np.uint8)

    upper = np.array([
        min(179, h + tolerance),
        min(255, s + tolerance),
        min(255, v + tolerance)
    ], dtype=np.uint8)

    return lower, upper


def print_debug(
    logger,
    debug_verbose: bool,
    win_name: str,
    avg_fps: float,
    latest_contours: List,
    max_targets: int,
    current_target_idx: int,
    image_queue_size: int,
    image_queue_maxsize: int,
    **kwargs
):
    """Print debug information with rate limiting."""
    global last_debug_update
    if 'last_debug_update' not in globals():
        last_debug_update = 0
    if time.time() - last_debug_update < 0.2:
        return
    debug_lines = [
        f"Tracker Status @ {time.strftime('%H:%M:%S')}",
        f"Window: {win_name}",
        f"FPS: {avg_fps:.1f}",
        f"Contours: {len(latest_contours)}/{max_targets}",
        f"Current Target: {current_target_idx + 1 if latest_contours else 0}",
        f"Queue Size: {image_queue_size}/{image_queue_maxsize}",
    ]
    for key, value in kwargs.items():
        debug_lines.append(f"{key}: {value}")
    if debug_verbose:
        print("\033[2J\033[H", end="")
        print("\n".join(debug_lines))
    last_debug_update = time.time()