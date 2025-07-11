"""
window_color_tracker.py
Track a dark‑grey colour inside a specific window and visualize it.

Dependencies:
    pip install pywin32 mss opencv-python numpy
"""

import time
from typing import List, Tuple

import cv2
import numpy as np
import mss

# ─── pywin32 imports ──────────────────────────────────────────────────────
import win32con
import win32gui


# ─── CONFIGURATION ────────────────────────────────────────────────────────
WINDOW_TITLE       = "BloodStrike"      # part or full window title
TARGET_COLOURS_BGR = [(0x2A, 0x2A, 0x2A)]
TOLERANCE          = 15                  # ± per‑channel tolerance
MIN_BLOB_AREA      = 30                  # px²
SHOW_FPS           = False


# ─── WINDOW HELPERS ───────────────────────────────────────────────────────
def find_window(title: str) -> int:
    """Return HWND of the first top‑level window containing `title`."""
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        return hwnd

    # Fallback: iterate all windows and match substring (case‑insensitive)
    def _enum_callback(h, result):
        if title.lower() in win32gui.GetWindowText(h).lower():
            result.append(h)
    matches = []
    win32gui.EnumWindows(_enum_callback, matches)
    if matches:
        return matches[0]
    raise RuntimeError(f'Window with title "{title}" not found.')


def get_client_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """
    Return (left, top, width, height) **in screen coordinates**
    for the *client area* of the given HWND.
    """
    # Get client rect (0,0‑relative) then convert to screen
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top   = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return left, top, right - left, bottom - top


def focus_window(hwnd: int) -> None:
    """Optionally bring the window to front (avoids capturing behind other apps)."""
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except win32gui.error:
        pass  # e.g. running without foreground permission


# ─── IMAGE UTILITIES ──────────────────────────────────────────────────────
def build_mask(frame_bgr: np.ndarray,
               colours: List[Tuple[int, int, int]],
               tol: int) -> np.ndarray:
    """Binary mask: 1 where any pixel ≈ any target colour."""
    masks = []
    for b, g, r in colours:
        lower = np.array([max(0,   b - tol),
                          max(0,   g - tol),
                          max(0,   r - tol)], dtype=np.uint8)
        upper = np.array([min(255, b + tol),
                          min(255, g + tol),
                          min(255, r + tol)], dtype=np.uint8)
        masks.append(cv2.inRange(frame_bgr, lower, upper))
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    return mask


# ─── MAIN LOOP ────────────────────────────────────────────────────────────
def main() -> None:
    hwnd = find_window(WINDOW_TITLE)
    focus_window(hwnd)                      # optional

    with mss.mss() as sct:
        cx = cy = None                      # client‑area centre

        while True:
            start = time.time()

            # Refresh region every loop (window can move)
            left, top, width, height = get_client_rect(hwnd)
            region = {"left": left, "top": top, "width": width, "height": height}

            # Capture
            grab = sct.grab(region)         # BGRA
            frame = np.array(grab)[:, :, :3]  # -> BGR

            if cx is None:
                cx, cy = width // 2, height // 2

            # Build mask & cleanup
            mask = build_mask(frame, TARGET_COLOURS_BGR, TOLERANCE)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8), iterations=1)

            # Detect blobs
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Draw overlay (centres are relative *to the frame*)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_BLOB_AREA:
                    continue
                M = cv2.moments(cnt)
                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])
                cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)
                cv2.line(frame, (cx, cy), (px, py), (0, 255, 0), 1)

            # Show windows
            title = f"Colour Tracker – {WINDOW_TITLE}"
            if SHOW_FPS:
                fps = 1 / max(time.time() - start, 1e-6)
                title += f"  |  {fps:4.1f} FPS"
            cv2.imshow(title, frame)
            cv2.imshow("Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
