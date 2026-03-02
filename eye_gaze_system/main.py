"""Main entry point for the eye gaze control system."""

import cv2
import pyautogui
from typing import Optional, Tuple, Dict, Any
import sys
import os

try:
    from .config import Config, get_default_config, merge_config
    from .camera import Camera
    from .face_landmarks import FaceLandmarkDetector
    from .iris_tracker import IrisTracker
    from .gaze_estimator import GazeEstimator
    from .blink_detector import BlinkDetector
    from .calibration import Calibration
    from .filters.kalman_filter import KalmanFilter
    from .filters.gaussian_outlier_filter import GaussianOutlierFilter
    from .filters.one_euro_filter import OneEuroFilter
    from .filters.weighted_average_filter import WeightedAverageFilter
    from .tracking_state import TrackingState
    from .cursor_control import CursorController
    from .gaze_calibration import GazeCalibration, CalibrationSession
    from .utils import map_to_screen, clamp_screen_coordinates
    from .filters.gaze_processor import GazeProcessor
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eye_gaze_system.config import Config, get_default_config, merge_config
    from eye_gaze_system.camera import Camera
    from eye_gaze_system.face_landmarks import FaceLandmarkDetector
    from eye_gaze_system.iris_tracker import IrisTracker
    from eye_gaze_system.gaze_estimator import GazeEstimator
    from eye_gaze_system.blink_detector import BlinkDetector
    from eye_gaze_system.calibration import Calibration
    from eye_gaze_system.filters.kalman_filter import KalmanFilter
    from eye_gaze_system.filters.gaussian_outlier_filter import GaussianOutlierFilter
    from eye_gaze_system.filters.one_euro_filter import OneEuroFilter
    from eye_gaze_system.filters.weighted_average_filter import WeightedAverageFilter
    from eye_gaze_system.tracking_state import TrackingState
    from eye_gaze_system.cursor_control import CursorController
    from eye_gaze_system.gaze_calibration import GazeCalibration, CalibrationSession
    from eye_gaze_system.utils import map_to_screen, clamp_screen_coordinates
    from eye_gaze_system.filters.gaze_processor import GazeProcessor


class EyeGazeSystem:
    """Main eye gaze control system. Accepts config dictionary for all hyperparameters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with config. Pass dict to override defaults.

        Args:
            config: Optional config dict. Missing keys use defaults from get_default_config().
        """
        self.config = Config(config)
        cfg = self.config._config

        # Core components
        self.face_detector = FaceLandmarkDetector(refine_landmarks=cfg["refine_landmarks"])
        self.iris_tracker = IrisTracker(
            self.face_detector,
            use_both_eyes=cfg["use_both_eyes"],
            enable_fallback=cfg["enable_fallback"],
        )
        self.gaze_estimator = GazeEstimator(
            self.iris_tracker,
            self.face_detector,
            screen_distance=cfg["screen_distance"],
            screen_width=cfg["screen_width"],
            screen_height=cfg["screen_height"],
        )
        self.blink_detector = BlinkDetector(
            self.face_detector,
            ear_threshold=cfg["ear_threshold"],
            consecutive_frames=cfg["consecutive_frames"],
            long_blink_frames=cfg["long_blink_frames"],
            debounce_time=cfg["debounce_time"],
        )
        self.calibration = Calibration(calibration_file=cfg["calibration_file"])

        if cfg["use_calibration"]:
            self.calibration.load()

        # Filters
        self.use_filter = cfg["use_filter"]
        self.filter_type = (cfg["filter_type"] or "kalman").lower() if self.use_filter else None

        if self.use_filter:
            self.outlier_filter = GaussianOutlierFilter(
                window_size=cfg["outlier_window_size"],
                std_threshold=cfg["outlier_std_threshold"],
                use_kernel_weighting=cfg["outlier_use_kernel_weighting"],
                kernel_sigma=cfg["outlier_kernel_sigma"],
            )
            self._init_smoothing_filter(cfg)

        # Screen
        self.screen_w, self.screen_h = pyautogui.size()

        # Tracking state
        self.tracking_state = TrackingState(
            confidence_threshold=cfg["confidence_threshold"],
            freeze_threshold=cfg["freeze_threshold"],
            max_freeze_frames=cfg["max_freeze_frames"],
            stability_window=cfg["stability_window"],
            max_frame_drop=cfg.get("max_frame_drop", 30),
        )

        # Cursor
        self.cursor_controller = CursorController(
            enable_interpolation=cfg["enable_interpolation"],
            interpolation_steps=cfg["interpolation_steps"],
            disable_failsafe=cfg.get("disable_failsafe", True),
            min_move_px=cfg.get("cursor_min_move_px", 8),
        )

        # 9-point gaze calibration — accurate nonlinear iris→screen mapping
        self.gaze_calib = GazeCalibration(
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            invert_x=cfg.get("iris_invert_x", True),
            gain_x=float(cfg.get("relative_iris_gain_x", 39.5)),
            gain_y=float(cfg.get("relative_iris_gain_y", 72.8)),
        )
        self._calib_session: Optional[CalibrationSession] = None

        # Warmup centre (resting iris position, used when no 9-point calib exists)
        self._gaze_centre: Optional[Tuple[float, float]] = None
        self._gaze_centre_samples: list = []
        self._GAZE_CENTRE_WARMUP = 60

        # State
        self.last_cursor_position: Optional[Tuple[int, int]] = None
        self.current_fps = 0.0
        self.debug_mode = False
        self.debug_data: Dict[str, Any] = {"raw_gaze": None, "filtered_gaze": None, "ear": None}
        self._debug_frame_count = 0
        self._process_frame_count = 0
        self._last_relative_iris: Optional[Tuple[float, float]] = None
        
        # New Advanced Edge Mapping, Padding, and Filtering Model
        self.advanced_processor = GazeProcessor()

    def _init_smoothing_filter(self, cfg: Dict[str, Any]):
        ft = self.filter_type
        if ft == "kalman":
            self.smoothing_filter = KalmanFilter(
                process_noise=cfg["kalman_process_noise"],
                measurement_noise=cfg["kalman_measurement_noise"],
                dt=cfg["kalman_dt"],
            )
        elif ft == "one_euro":
            self.smoothing_filter = OneEuroFilter(
                min_cutoff=cfg["one_euro_min_cutoff"],
                beta=cfg["one_euro_beta"],
                d_cutoff=cfg["one_euro_d_cutoff"],
            )
        elif ft == "weighted":
            self.smoothing_filter = WeightedAverageFilter(
                window_size=cfg["weighted_window_size"],
                alpha=cfg["weighted_alpha"],
            )
        else:
            self.filter_type = "kalman"
            self.smoothing_filter = KalmanFilter(
                process_noise=cfg["kalman_process_noise"],
                measurement_noise=cfg["kalman_measurement_noise"],
                dt=cfg["kalman_dt"],
            )

    def process_frame(
        self,
        frame: cv2.typing.MatLike,
        draw_visualization: bool = True,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[list]]:
        """
        Process frame through pipeline. Returns (screen_coords, landmarks).
        Landmarks are returned to avoid duplicate detection for blink handling.
        """
        self.tracking_state.check_frame_drop()
        self._process_frame_count += 1

        if self.debug_mode:
            self.debug_data = {"raw_gaze": None, "filtered_gaze": None, "ear": None}

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None, None
        rgb_frame = self.face_detector.frame_to_rgb(frame)
        landmarks, confidence = self.face_detector.detect(rgb_frame)

        min_conf = self.config.get("min_confidence", 0.2)
        if confidence < min_conf:
            return None, landmarks

        self.tracking_state.update_confidence(confidence)

        # Warmup: don't freeze cursor for first 90 frames so tracking can establish
        if self._process_frame_count >= 90 and self.tracking_state.should_freeze_cursor():
            if self.tracking_state.get_frozen_position() and self.last_cursor_position:
                return self.last_cursor_position, landmarks
            return None, landmarks

        if landmarks is None:
            return None, None

        # Relative iris position in eye socket (0-1): responds to eyeball movement, not head
        relative_iris = self.face_detector.get_left_eye_relative_iris_position(landmarks)
        used_fallback = False
        if self._last_relative_iris is not None:
            lx, ly = self._last_relative_iris
            if abs(iris_x - lx) < self.config.get("gaze_dead_zone", 0.0015) and \
                abs(iris_y - ly) < self.config.get("gaze_dead_zone", 0.0015):
                iris_x, iris_y = lx, ly
            self._last_relative_iris = (iris_x, iris_y)
        if relative_iris is None:
            # Fallback: absolute iris -> screen so cursor still moves
            iris_center_2d = self.face_detector.get_iris_center_2d(
                landmarks, frame_w, frame_h, eye="left"
            )
            if iris_center_2d is None:
                return None, landmarks
            used_fallback = True
            ix, iy = iris_center_2d
            screen_x_direct = int(ix / frame_w * self.screen_w)
            screen_y_direct = int(iy / frame_h * self.screen_h)
        else:
            # relative_iris = raw (iris_x, iris_y) from landmark 473
            iris_x, iris_y = relative_iris

            # ── Calibration session active: feed frame, don't move cursor ──
            if self._calib_session is not None:
                done, frame = self._calib_session.update(frame, (iris_x, iris_y))
                if done:
                    # Session finished — update gaze_calib centre too
                    self.gaze_calib = self._calib_session.calib
                    self._gaze_centre = (self.gaze_calib.cx, self.gaze_calib.cy)
                    self._calib_session = None
                    self.advanced_processor.prev_raw = None
                    self.advanced_processor.prev_smooth = None
                return None, landmarks

            # ── If 9-point calibration exists, use it directly ─────────────
            if self.gaze_calib.calibrated:
                screen_x_direct, screen_y_direct = self.gaze_calib.map(iris_x, iris_y)

            else:
                # ── Fallback: warmup centre + linear gain ──────────────────
                if self._gaze_centre is None:
                    self._gaze_centre_samples.append((iris_x, iris_y))
                    if len(self._gaze_centre_samples) >= self._GAZE_CENTRE_WARMUP:
                        cx = sum(s[0] for s in self._gaze_centre_samples) / len(self._gaze_centre_samples)
                        cy = sum(s[1] for s in self._gaze_centre_samples) / len(self._gaze_centre_samples)
                        self._gaze_centre = (cx, cy)
                        self.gaze_calib.cx = cx
                        self.gaze_calib.cy = cy
                        print(f"[CogniGaze] Gaze centre locked: iris_x={cx:.4f}  iris_y={cy:.4f}")
                        print("[CogniGaze] Press C to run 9-point calibration for accurate mapping.")
                    screen_x_direct = self.screen_w // 2
                    screen_y_direct = self.screen_h // 2
                else:
                    screen_x_direct, screen_y_direct = self.gaze_calib.map(iris_x, iris_y)

        # ── Apply Advanced Edge Compensation & Smoothing ─────────────
        # Normalize into [0, 1] range for GazeProcessor
        norm_x = screen_x_direct / self.screen_w
        norm_y = screen_y_direct / self.screen_h
        
        sm_norm_x, sm_norm_y = self.advanced_processor.map_and_smooth(norm_x, norm_y)
        
        # Denormalize back to screen pixels
        screen_x_direct = int(round(sm_norm_x * self.screen_w))
        screen_y_direct = int(round(sm_norm_y * self.screen_h))

        screen_x_direct, screen_y_direct = clamp_screen_coordinates(
            screen_x_direct, screen_y_direct, self.screen_w, self.screen_h
        )

        self.last_cursor_position = (screen_x_direct, screen_y_direct)
        if draw_visualization:
            if landmarks:
                self._draw_visualization(frame, landmarks, frame_w, frame_h)
            self._draw_tracking_state(frame, frame_w, frame_h, self.current_fps)
            if self.debug_mode:
                self._draw_debug_overlay(frame, frame_w, frame_h)
        return (screen_x_direct, screen_y_direct), landmarks

    def _draw_visualization(
        self,
        frame: cv2.typing.MatLike,
        landmarks,
        frame_w: int,
        frame_h: int,
    ):
        left_iris = self.face_detector.get_left_iris_landmarks(landmarks)
        if left_iris:
            for lm in left_iris:
                x, y = int(lm[0] * frame_w), int(lm[1] * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        right_iris = self.face_detector.get_right_iris_landmarks(landmarks)
        if right_iris:
            for lm in right_iris:
                x, y = int(lm[0] * frame_w), int(lm[1] * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for label, eye in [("left", "left"), ("right", "right")]:
            center = self.face_detector.get_iris_center_2d(landmarks, frame_w, frame_h, eye)
            if center:
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        left_eye = self.face_detector.get_left_eye_landmarks(landmarks)
        if left_eye:
            for lm in left_eye:
                x, y = int(lm[0] * frame_w), int(lm[1] * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    def _draw_tracking_state(
        self,
        frame: cv2.typing.MatLike,
        frame_w: int,
        frame_h: int,
        fps: float = 0.0,
    ):
        conf = self.tracking_state.current_confidence
        is_frozen = self.tracking_state.is_frozen
        is_occluded = self.tracking_state.is_partially_occluded()
        is_moving = self.tracking_state.is_head_moving()

        if is_frozen:
            color = (0, 0, 255)
            status_text = "FROZEN"
        elif is_occluded:
            color = (0, 165, 255)
            status_text = "PARTIAL"
        elif is_moving:
            color = (0, 255, 255)
            status_text = "MOVING"
        else:
            color = (0, 255, 0)
            status_text = "TRACKING"

        # Show warmup countdown if gaze centre not yet locked
        if self._gaze_centre is None and not self.gaze_calib.calibrated:
            samples = len(self._gaze_centre_samples)
            warmup_pct = int(samples / self._GAZE_CENTRE_WARMUP * 100)
            warmup_text = f"CALIBRATING... {warmup_pct}%  Look straight at screen"
            cv2.putText(frame, warmup_text, (10, frame_h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        status_line = f"Status: {status_text} | Conf: {conf:.2f}"
        if fps > 0:
            status_line += f" | FPS: {fps:.1f}"
        if self.gaze_calib.calibrated:
            status_line += "  [C]=recalibrate  [R]=reset"
        elif self._gaze_centre is not None:
            status_line += "  [C]=9pt-calibrate  [R]=reset"
        cv2.putText(frame, status_line, (10, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bar_w, bar_h = 200, 10
        bx, by = 10, frame_h - 30
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
        fill = int(bar_w * conf)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 1)

    def _draw_debug_overlay(self, frame: cv2.typing.MatLike, frame_w: int, frame_h: int):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 30
        line_h = 22
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += line_h
        cv2.putText(frame, f"Confidence: {self.tracking_state.current_confidence:.3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += line_h
        ear = self.debug_data.get("ear")
        ear_str = f"{ear:.4f}" if ear is not None else "N/A"
        cv2.putText(frame, f"EAR: {ear_str}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += line_h

        raw = self.debug_data.get("raw_gaze")
        if raw is not None:
            rx, ry = int(raw[0] * frame_w), int(raw[1] * frame_h)
            cv2.circle(frame, (rx, ry), 8, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw: ({raw[0]:.3f}, {raw[1]:.3f})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Raw: N/A", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        y += line_h

        flt = self.debug_data.get("filtered_gaze")
        if flt is not None:
            fx, fy = int(flt[0] * frame_w), int(flt[1] * frame_h)
            cv2.circle(frame, (fx, fy), 6, (255, 0, 255), 2)
            cv2.putText(frame, f"Filtered: ({flt[0]:.3f}, {flt[1]:.3f})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        else:
            cv2.putText(frame, "Filtered: N/A", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.putText(frame, "[D] Debug | [Q] Quit", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def run(self, camera_index: Optional[int] = None, target_fps: Optional[float] = None):
        import time

        cam_idx = camera_index if camera_index is not None else self.config.get("camera_index", 0)
        fps = target_fps if target_fps is not None else self.config.get("target_fps", 30.0)

        cam_cfg = {
            "flip_horizontal": self.config.get("camera_flip_horizontal", True),
            "use_threading": self.config.get("camera_use_threading", True),
            "buffer_size": self.config.get("camera_buffer_size", 2),
        }

        with Camera(camera_index=cam_idx, **cam_cfg) as camera:
            if not camera.cap:
                print("Failed to initialize camera")
                return

            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0   # default is 0.1s — 100ms lag per moveTo call
            calib_status = "9-pt calibration loaded [OK]" if self.gaze_calib.calibrated else "No calibration — press C to calibrate"
            print(f"Eye Gaze System started. Press q=quit, d=debug, c=calibrate, r=reset centre.")
            print(f"Screen: {self.screen_w}x{self.screen_h}  |  {calib_status}")

            frame_time = 1.0 / fps
            last_time = time.time()
            frame_count = 0
            fps_start = time.time()

            while True:
                # No artificial sleep — let camera + MediaPipe set the natural pace
                last_time = time.time()

                ok, frame = camera.read_frame()
                if not ok:
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    self.current_fps = 30.0 / elapsed if elapsed > 0 else 0
                    fps_start = time.time()
                elif frame_count == 1:
                    self.current_fps = 0.0

                screen_coords, landmarks = self.process_frame(frame, draw_visualization=True)

                if screen_coords:
                    self.cursor_controller.move_to(screen_coords[0], screen_coords[1])

                if landmarks:
                    should_click, click_type = self.blink_detector.should_trigger_click(landmarks, eye="left")
                    if should_click:
                        if click_type == "left":
                            pyautogui.click()
                        elif click_type == "right":
                            pyautogui.rightClick()

                cv2.imshow("Cognigaze – Hybrid Predictive Gaze Interaction System", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    self.debug_mode = not self.debug_mode
                    print("Debug overlay ON" if self.debug_mode else "Debug overlay OFF")
                elif key == ord("c"):
                    # Start 9-point gaze calibration
                    print("[CogniGaze] Starting 9-point calibration. Look at each dot and press SPACE.")
                    self._calib_session = CalibrationSession(
                        self.gaze_calib, self.screen_w, self.screen_h
                    )
                    self.advanced_processor.prev_raw = None
                    self.advanced_processor.prev_smooth = None
                elif key == ord(" ") and self._calib_session is not None:
                    self._calib_session.confirm_point()
                elif key == ord("r"):
                    # Reset gaze centre — look straight at screen and press r
                    self._gaze_centre = None
                    self._gaze_centre_samples = []
                    self.advanced_processor.prev_raw = None
                    self.advanced_processor.prev_smooth = None
                    self._last_relative_iris = None
                    print("[CogniGaze] Gaze centre RESET — hold still for 2 seconds to recalibrate.")

            cv2.destroyAllWindows()

    def run_calibration(self, camera_index: Optional[int] = None):
        cam_idx = camera_index if camera_index is not None else self.config.get("camera_index", 0)
        cam_cfg = {
            "flip_horizontal": self.config.get("camera_flip_horizontal", True),
            "use_threading": False,
        }

        with Camera(camera_index=cam_idx, **cam_cfg) as camera:
            if not camera.cap:
                print("Failed to initialize camera")
                return

            print("Calibration mode started.")
            print("Look at each red dot and press SPACE when ready.")
            print("Press 'q' to quit, 'r' to reset calibration.")

            self.calibration.start_calibration()

            while True:
                ok, frame = camera.read_frame()
                if not ok:
                    continue

                frame_h, frame_w = frame.shape[:2]
                self.calibration.draw_calibration_grid(frame, frame_w, frame_h)

                if self.calibration.get_current_calibration_point() is None:
                    if self.calibration.calibrated:
                        print("Calibration complete! Saving...")
                        if self.calibration.save():
                            print("Calibration saved successfully!")
                        else:
                            print("Error saving calibration.")
                    else:
                        print("Calibration failed. Please try again.")
                    break

                rgb_frame = self.face_detector.frame_to_rgb(frame)
                landmarks, _ = self.face_detector.detect(rgb_frame)

                if landmarks:
                    gaze_point = self.gaze_estimator.estimate_gaze(
                        landmarks,
                        frame=frame,
                        frame_width=frame_w,
                        frame_height=frame_h,
                    )
                    if gaze_point:
                        self.calibration.collect_gaze_sample(gaze_point)
                        gx, gy = int(gaze_point[0] * frame_w), int(gaze_point[1] * frame_h)
                        cv2.circle(frame, (gx, gy), 5, (255, 255, 0), -1)

                cv2.imshow("Calibration Mode", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.calibration.reset()
                    self.calibration.start_calibration()
                    print("Calibration reset.")
                elif key == ord(" "):
                    if self.calibration.is_collecting():
                        self.calibration.finish_current_point()
                        print(f"Point {self.calibration.current_point_index} recorded.")

            cv2.destroyAllWindows()


def main():
    """Entry point. Use config dict to customize."""
    args = sys.argv[1:]
    calibrate = "--calibrate" in args
    if calibrate:
        args.remove("--calibrate")
    debug_direct = "--debug-direct-cursor" in args
    if debug_direct:
        args.remove("--debug-direct-cursor")

    camera_index = int(args[0]) if len(args) > 0 else 0
    filter_type = (args[1] or "kalman").lower() if len(args) > 1 else "kalman"
    if filter_type not in ("kalman", "one_euro", "weighted"):
        filter_type = "kalman"

    config = merge_config({
        "camera_index": camera_index,
        "filter_type": filter_type,
        "use_calibration": True,
        "debug_direct_cursor": debug_direct,
    })

    system = EyeGazeSystem(config=config)

    if calibrate:
        print("Starting calibration mode...")
        system.run_calibration()
    else:
        print(f"Using {filter_type} filter")
        if system.calibration.calibrated:
            print("Calibration loaded successfully.")
        else:
            print("No calibration found. Run with --calibrate to calibrate.")
        system.run()


if __name__ == "__main__":
    main()