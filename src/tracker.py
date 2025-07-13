import threading
import queue
import time
from typing import List, Tuple, Optional, Dict
from collections import deque
from PIL import Image
import numpy as np
import cv2 as cv
import win32gui
import win32api
import win32con
import ctypes
import logging
from mss import mss
from utils import precompute_hsv, print_debug, MatLike


class Tracker:
    def __init__(
        self,
        win_name: str,
        target_color: Tuple[int, int, int],
        fov: int,
        sens: int,
        offset_head: float,
        display_fps: bool,
        display_contours: bool,
        lead: int,
        min_area: int,
        max_targets: int,
        process_scale: float,
        target_switch_interval: float,
        debug_verbose: bool,
        color_tolerance: int,
        smoothing_factor: float,
    ):
        """Initialize the Tracker with validated parameters."""
        self.hwnd = win32gui.FindWindow(None, win_name)
        if not self.hwnd:
            raise RuntimeError(f"Window '{win_name}' not found.")

        logging.basicConfig(
            filename="tracker.log",
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            filemode="w",
        )
        self.logger = logging.getLogger()
        self.win_name = win_name
        self.target_color = target_color
        self.fov = fov
        self.sens = sens
        self.offset_head = offset_head
        self.display_fps = display_fps
        self.display_contours = display_contours
        self.lead = lead
        self.min_area = min_area
        self.max_targets = max_targets
        self.process_scale = process_scale
        self.target_switch_interval = target_switch_interval
        self.debug_verbose = debug_verbose
        self.color_tolerance = color_tolerance
        self.smoothing_factor = smoothing_factor

        self.image_queue: queue.Queue[Tuple[Image.Image, dict]] = queue.Queue(maxsize=2)
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.latest_contours: List[Tuple[np.ndarray, float]] = []
        self.monitor = None
        self.target_histories: Dict[int, deque] = {}
        self.hsv_lower, self.hsv_upper = precompute_hsv(target_color, color_tolerance)
        self.current_target_idx = 0
        self.last_target_switch = time.time()
        self.avg_fps = 0
        self.last_delta_x = 0
        self.last_delta_y = 0
        self.calibration_point: Tuple[int, int] = (0, 0)
        self.toggle = False

    def calibrate_color(self):
        """Allow user to click on screen to set target color."""
        img, monitor = self.get_screen()
        if img is None:
            self.logger.error("Failed to capture screen for calibration")
            return
        img_np = np.array(img)
        cv.imshow("Calibration", img_np)

        def mouse_callback(event, x: int, y: int, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                self.calibration_point = (x, y)

        cv.setMouseCallback("Calibration", mouse_callback)
        while self.running.is_set():
            if self.calibration_point != (0, 0):
                break
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        x, y = self.calibration_point
        if x != 0 and y != 0:
            pixel_color = tuple(int(c) for c in img_np[y, x])
            self.target_color = (pixel_color[2], pixel_color[1], pixel_color[0])
            self.hsv_lower, self.hsv_upper = precompute_hsv(
                self.target_color, self.color_tolerance
            )
            self.logger.info(f"New target color: {self.target_color}")
            if self.debug_verbose:
                print(f"New target color: {self.target_color}")
        cv.destroyWindow("Calibration")

    def get_screen(self) -> Tuple[Optional[Image.Image], Optional[dict]]:
        """Capture screen region around the target window."""
        try:
            if not win32gui.IsWindow(self.hwnd):
                self.hwnd = win32gui.FindWindow(None, self.win_name)
                if not self.hwnd:
                    self.logger.error(f"Window '{self.win_name}' no longer exists.")
                    return None, None

            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                time.sleep(0.1)

            client_rect = win32gui.GetClientRect(self.hwnd)
            client_x, client_y = win32gui.ClientToScreen(self.hwnd, (0, 0))
            win_width = client_rect[2] - client_rect[0]
            win_height = client_rect[3] - client_rect[1]
            center_x = client_x + win_width // 2
            center_y = client_y + win_height // 2
            fov_half = self.fov // 2

            monitor = {
                "top": max(client_y, center_y - fov_half),
                "left": max(client_x, center_x - fov_half),
                "width": min(client_x + win_width, center_x + fov_half)
                - max(client_x, center_x - fov_half),
                "height": min(client_y + win_height, center_y + fov_half)
                - max(client_y, center_y - fov_half),
            }

            if monitor["width"] <= 0 or monitor["height"] <= 0:
                self.logger.error(f"Invalid dimensions: {monitor}")
                return None, None

            with mss() as sct:
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                if self.process_scale != 1.0:
                    img_np = np.array(img)
                    img_np = cv.resize(
                        img_np,
                        (
                            int(img.width * self.process_scale),
                            int(img.height * self.process_scale),
                        ),
                        interpolation=cv.INTER_LANCZOS4,
                    )
                    img = Image.fromarray(img_np)
                print_debug(
                    self.logger,
                    self.debug_verbose,
                    self.win_name,
                    self.avg_fps,
                    self.latest_contours,
                    self.max_targets,
                    self.current_target_idx,
                    self.image_queue.qsize(),
                    self.image_queue.maxsize,
                    Screen_Capture="Success",
                    Region=f"{monitor['width']}x{monitor['height']} @ ({monitor['left']}, {monitor['top']})",
                )
                return img, monitor
        except Exception as e:
            self.logger.error(f"Error capturing screen: {e}")
            if self.debug_verbose:
                print(f"Error capturing screen: {e}")
            return None, None

    def detect_contours(self, img_np: MatLike) -> List[Tuple[np.ndarray, float]]:
        """Detect contours using combined HSV masking and Canny edge highlighting."""
        if not isinstance(img_np, np.ndarray) or img_np.size == 0:
            return []

        img_np = np.asarray(img_np, dtype=np.uint8)
        if img_np.shape[0] < 10 or img_np.shape[1] < 10:
            return []

        try:
            # Convert to HSV and apply color mask
            hsv = cv.cvtColor(img_np, cv.COLOR_RGB2HSV)
            mask_color = cv.inRange(hsv, self.hsv_lower, self.hsv_upper)

            # Convert to grayscale and apply Canny edge detection
            gray = cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
            edges = cv.Canny(gray, 50, 150)

            # Combine both: keep only edges that match the color
            combined_mask = cv.bitwise_and(edges, mask_color)

            # Close small gaps
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv.morphologyEx(
                combined_mask, cv.MORPH_CLOSE, kernel, iterations=2
            )

            # Find contours
            contours, _ = cv.findContours(
                combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            valid_contours = [
                (c, cv.contourArea(c))
                for c in contours
                if cv.contourArea(c) >= self.min_area
            ]
            return sorted(valid_contours, key=lambda x: x[1], reverse=True)[
                : self.max_targets
            ]

        except cv.error as e:
            if self.debug_verbose:
                print(f"[ERROR] Contour detection failed: {e}")
            return []

    def capture_thread(self):
        """Capture screenshots and add to queue."""
        last_capture = time.time()
        target_fps = 60
        min_sleep = 1.0 / target_fps
        error_backoff = 0.1
        max_backoff = 1.0
        while self.running.is_set():
            img, monitor = self.get_screen()
            if img is None or monitor is None:
                time.sleep(error_backoff)
                error_backoff = min(error_backoff * 2, max_backoff)
                continue
            error_backoff = 0.1
            with self.lock:
                self.monitor = monitor
            try:
                self.image_queue.put_nowait((img, monitor))
            except queue.Full:
                print_debug(
                    self.logger,
                    self.debug_verbose,
                    self.win_name,
                    self.avg_fps,
                    self.latest_contours,
                    self.max_targets,
                    self.current_target_idx,
                    self.image_queue.qsize(),
                    self.image_queue.maxsize,
                    Queue_Status="Full, skipping frame",
                )
            elapsed = time.time() - last_capture
            time.sleep(max(0.001, min_sleep - elapsed))
            last_capture = time.time()

    def process_thread(self):
        """Process images and control mouse for target tracking."""
        last_time = time.time()
        frame_count = 0
        fps_update_interval = 0.5
        last_fps_update = last_time

        while self.running.is_set():
            try:
                img, _ = self.image_queue.get(timeout=0.1)
                img_np = np.array(img, dtype=np.uint8)

                with self.lock:
                    self.latest_contours = self.detect_contours(img_np)
                    if self.latest_contours and self.current_target_idx >= len(
                        self.latest_contours
                    ):
                        self.current_target_idx = 0
                        self.last_target_switch = time.time()

                if self.latest_contours:
                    img_center = (img_np.shape[1] / 2, img_np.shape[0] / 2)
                    sorted_contours = sorted(
                        self.latest_contours,
                        key=lambda c: (
                            (
                                cv.boundingRect(c[0])[0]
                                + cv.boundingRect(c[0])[2] / 2
                                - img_center[0]
                            )
                            ** 2
                            + (
                                cv.boundingRect(c[0])[1]
                                + cv.boundingRect(c[0])[3] / 2
                                - img_center[1]
                            )
                            ** 2
                        )
                        ** 0.5,
                    )
                    with self.lock:
                        self.latest_contours = sorted_contours

                if self.display_contours and self.latest_contours:
                    for i, (contour, _) in enumerate(self.latest_contours):
                        cv.drawContours(
                            img_np,
                            [contour],
                            -1,
                            (
                                (0, 255, 0)
                                if i == self.current_target_idx
                                else self.target_color
                            ),
                            thickness=cv.FILLED,
                        )

                current_time = time.time()
                frame_count += 1
                if current_time - last_fps_update >= fps_update_interval:
                    self.avg_fps = frame_count / max(
                        current_time - last_fps_update, 1e-5
                    )
                    frame_count = 0
                    last_fps_update = current_time

                if self.display_fps:
                    cv.putText(
                        img_np,
                        f"FPS: {self.avg_fps:.1f}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        img_np,
                        f"Target: {self.current_target_idx}/{len(self.latest_contours)}",
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )

                mouse_moved = False
                with self.lock:
                    if (
                        self.latest_contours
                        and self.toggle
                        and (
                            win32api.GetAsyncKeyState(win32con.VK_LBUTTON) < 0
                            or win32api.GetAsyncKeyState(win32con.VK_RBUTTON) < 0
                        )
                        and self.monitor
                    ):
                        if (
                            current_time - self.last_target_switch
                            >= self.target_switch_interval
                        ):
                            self.current_target_idx = (
                                self.current_target_idx + 1
                            ) % len(self.latest_contours)
                            self.last_target_switch = current_time
                            self.logger.debug(
                                f"Switched to target {self.current_target_idx}"
                            )

                        for i, (contour, _) in enumerate(self.latest_contours):
                            x, y, w, h = cv.boundingRect(contour)
                            center_x = x + w // 2
                            head_y = y + int(h * self.offset_head)
                            self.target_histories.setdefault(i, deque(maxlen=5)).append(
                                (center_x, head_y)
                            )

                        if self.current_target_idx < len(self.latest_contours):
                            contour, _ = self.latest_contours[self.current_target_idx]
                            x, y, w, h = cv.boundingRect(contour)
                            center_x = x + w // 2
                            head_y = y + int(h * self.offset_head)
                            time_diff = max(current_time - last_time, 1e-5)
                            history = self.target_histories.get(
                                self.current_target_idx, deque(maxlen=5)
                            )

                            pred_x, pred_y = center_x, head_y
                            if len(history) >= self.lead:
                                prev_pos = history[-self.lead]
                                velocity_x = (center_x - prev_pos[0]) / time_diff
                                velocity_y = (head_y - prev_pos[1]) / time_diff
                                pred_x += velocity_x * time_diff
                                pred_y += velocity_y * time_diff

                            cursor_pos = win32api.GetCursorPos()
                            to = (
                                int(pred_x / self.process_scale + self.monitor["left"]),
                                int(pred_y / self.process_scale + self.monitor["top"]),
                            )
                            delta_x = int((to[0] - cursor_pos[0]) / self.sens)
                            delta_y = int((to[1] - cursor_pos[1]) / self.sens)

                            smoothed_x = (
                                self.smoothing_factor * delta_x
                                + (1 - self.smoothing_factor) * self.last_delta_x
                            )
                            smoothed_y = (
                                self.smoothing_factor * delta_y
                                + (1 - self.smoothing_factor) * self.last_delta_y
                            )
                            self.last_delta_x, self.last_delta_y = (
                                smoothed_x,
                                smoothed_y,
                            )

                            if abs(smoothed_x) > 0.1 or abs(smoothed_y) > 0.1:
                                ctypes.windll.user32.mouse_event(
                                    win32con.MOUSEEVENTF_MOVE,
                                    int(smoothed_x),
                                    int(smoothed_y),
                                    0,
                                    0,
                                )
                                mouse_moved = True

                print_debug(
                    self.logger,
                    self.debug_verbose,
                    self.win_name,
                    self.avg_fps,
                    self.latest_contours,
                    self.max_targets,
                    self.current_target_idx,
                    self.image_queue.qsize(),
                    self.image_queue.maxsize,
                    Mouse_Move=(
                        f"dx={int(smoothed_x)}, dy={int(smoothed_y)}"
                        if mouse_moved
                        else "None"
                    ),
                    Target_Switch=(
                        f"Target {self.current_target_idx + 1}"
                        if mouse_moved
                        else "None"
                    ),
                    Toggle=f"Toggle: {self.toggle}",
                )

                if self.display_contours or self.display_fps:
                    cv.imshow("Processed Roblox", img_np)
                    if (
                        cv.waitKey(1) & 0xFF == ord("q")
                        or win32api.GetAsyncKeyState(win32con.VK_F1) < 0
                    ):
                        self.running.clear()
                elif win32api.GetAsyncKeyState(win32con.VK_F1) < 0:
                    self.running.clear()

                last_time = current_time
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
                if self.debug_verbose:
                    print(f"Error in processing thread: {e}")
                time.sleep(0.1)

    def run(self):
        """Start the tracker threads."""
        try:
            threads = [
                threading.Thread(target=self.capture_thread, daemon=True),
                threading.Thread(target=self.process_thread, daemon=True),
                threading.Thread(target=self.keyboard, daemon=True),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            cv.destroyAllWindows()

    def keyboard(self):
        """Handle keyboard inputs for calibration and toggle."""
        while self.running.is_set():
            if win32api.GetAsyncKeyState(win32con.VK_F2) < 0:
                self.calibrate_color()
            if win32api.GetAsyncKeyState(win32con.VK_F3) < 0:
                self.toggle = not self.toggle
                self.logger.debug(f"Toggle state: {self.toggle}")
                time.sleep(0.5)
