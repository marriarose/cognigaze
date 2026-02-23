"""
9-point gaze calibration for CogniGaze.

Records iris position while user looks at each of 9 known screen locations,
then uses bilinear interpolation to map any iris position to screen coordinates.
This replaces the linear gain formula and fixes mapping inaccuracy.

Usage:
    Press C in the main window to enter calibration mode.
    Look at each red dot and press SPACE to record.
    Calibration auto-saves to disk and loads on next run.
"""

import numpy as np
import json
import os
from typing import Optional, Tuple, List


# 9 calibration points as (screen_x_frac, screen_y_frac) — fractions of screen size
# Arranged in a 3x3 grid with 10% margin from edges
CALIB_POINTS_FRAC = [
    (0.1, 0.1),  (0.5, 0.1),  (0.9, 0.1),   # top row:    left, centre, right
    (0.1, 0.5),  (0.5, 0.5),  (0.9, 0.5),   # middle row: left, centre, right
    (0.1, 0.9),  (0.5, 0.9),  (0.9, 0.9),   # bottom row: left, centre, right
]

CALIB_FILE = os.path.expanduser("~/.cognigaze_calib.json")
SAMPLES_PER_POINT = 30   # frames averaged per calibration point


class GazeCalibration:
    """
    Maps raw iris coordinates to screen coordinates using 9-point calibration.

    After calibration, self.map(iris_x, iris_y) → (screen_x, screen_y).
    Falls back to linear mapping if not calibrated.
    """

    def __init__(self, screen_w: int, screen_h: int,
                 calib_file: str = CALIB_FILE,
                 invert_x: bool = True,
                 gain_x: float = 39.5,
                 gain_y: float = 72.8):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.calib_file = calib_file
        self.invert_x = invert_x
        self.gain_x = gain_x
        self.gain_y = gain_y

        # Calibration data: list of (iris_x, iris_y, screen_x_frac, screen_y_frac)
        self.points: List[Tuple[float, float, float, float]] = []
        self.calibrated = False

        # Gaze centre (resting iris position = screen centre)
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        data = {
            "points": self.points,
            "cx": self.cx,
            "cy": self.cy,
            "invert_x": self.invert_x,
            "gain_x": self.gain_x,
            "gain_y": self.gain_y,
        }
        try:
            with open(self.calib_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[GazeCalib] Saved to {self.calib_file}")
        except Exception as e:
            print(f"[GazeCalib] Save failed: {e}")

    def load(self) -> bool:
        if not os.path.exists(self.calib_file):
            return False
        try:
            with open(self.calib_file) as f:
                data = json.load(f)
            self.points  = [tuple(p) for p in data.get("points", [])]
            self.cx       = data.get("cx")
            self.cy       = data.get("cy")
            self.invert_x = data.get("invert_x", self.invert_x)
            self.gain_x   = data.get("gain_x",   self.gain_x)
            self.gain_y   = data.get("gain_y",   self.gain_y)
            if len(self.points) == 9 and self.cx is not None:
                self.calibrated = True
                print(f"[GazeCalib] Loaded {len(self.points)}-point calibration.")
            return True
        except Exception as e:
            print(f"[GazeCalib] Load failed: {e}")
            return False

    def reset(self):
        self.points = []
        self.calibrated = False
        if os.path.exists(self.calib_file):
            os.remove(self.calib_file)

    # ── Mapping ───────────────────────────────────────────────────────────────

    def map(self, iris_x: float, iris_y: float) -> Tuple[int, int]:
        """Map raw iris position to screen coordinates."""
        if self.calibrated and len(self.points) == 9:
            return self._bilinear_map(iris_x, iris_y)
        return self._linear_map(iris_x, iris_y)

    def _linear_map(self, iris_x: float, iris_y: float) -> Tuple[int, int]:
        """Fallback: simple gain-based mapping using gaze centre."""
        if self.cx is None or self.cy is None:
            return self.screen_w // 2, self.screen_h // 2
        dx = -(iris_x - self.cx) if self.invert_x else (iris_x - self.cx)
        dy = iris_y - self.cy
        sx = int(max(0, min(self.screen_w - 1,  self.screen_w  * (0.5 + dx * self.gain_x))))
        sy = int(max(0, min(self.screen_h - 1, self.screen_h * (0.5 + dy * self.gain_y))))
        return sx, sy

    def _bilinear_map(self, iris_x: float, iris_y: float) -> Tuple[int, int]:
        """
        Bilinear interpolation across the 9 calibration points.

        The 9 points form a 3×3 grid in iris space. We find which of the
        4 cells the query point falls into, then bilinearly interpolate
        the 4 corner screen positions to get the output.
        """
        # Unpack calibration points into iris coords and screen fracs
        # Points are stored in row-major order: row0=[0,1,2], row1=[3,4,5], row2=[6,7,8]
        iris  = [(p[0], p[1]) for p in self.points]   # iris (x,y) for each point
        sfrac = [(p[2], p[3]) for p in self.points]   # screen fraction (sx,sy) for each point

        # Rows in iris-X space (3 columns per row)
        # row index 0=top, 1=mid, 2=bot (by screen_y_frac ordering)
        # We need to find which 2×2 cell contains (iris_x, iris_y)

        # Build sorted column/row reference values from calibration
        # Column iris_x values: average of column 0,1,2
        col_x = [
            (iris[0][0] + iris[3][0] + iris[6][0]) / 3,   # left column avg iris_x
            (iris[1][0] + iris[4][0] + iris[7][0]) / 3,   # mid column avg iris_x
            (iris[2][0] + iris[5][0] + iris[8][0]) / 3,   # right column avg iris_x
        ]
        row_y = [
            (iris[0][1] + iris[1][1] + iris[2][1]) / 3,   # top row avg iris_y
            (iris[3][1] + iris[4][1] + iris[5][1]) / 3,   # mid row avg iris_y
            (iris[6][1] + iris[7][1] + iris[8][1]) / 3,   # bot row avg iris_y
        ]

        # Find which cell — clamp to valid range
        def interp_t(val, lo, hi):
            if hi == lo:
                return 0.5
            return max(0.0, min(1.0, (val - lo) / (hi - lo)))

        # Determine column cell (0 = left-to-mid, 1 = mid-to-right)
        if iris_x <= col_x[1]:
            ci = 0
            tx = interp_t(iris_x, col_x[0], col_x[1])
        else:
            ci = 1
            tx = interp_t(iris_x, col_x[1], col_x[2])

        # Determine row cell (0 = top-to-mid, 1 = mid-to-bot)
        if iris_y <= row_y[1]:
            ri = 0
            ty = interp_t(iris_y, row_y[0], row_y[1])
        else:
            ri = 1
            ty = interp_t(iris_y, row_y[1], row_y[2])

        # 4 corners of the cell: indices into the 3×3 grid
        r0, r1 = ri,     ri + 1
        c0, c1 = ci,     ci + 1
        i00 = r0 * 3 + c0
        i10 = r0 * 3 + c1
        i01 = r1 * 3 + c0
        i11 = r1 * 3 + c1

        # Bilinear interpolation of screen fracs
        def bilerp(f00, f10, f01, f11, tx, ty):
            top = f00 * (1 - tx) + f10 * tx
            bot = f01 * (1 - tx) + f11 * tx
            return top * (1 - ty) + bot * ty

        sx_frac = bilerp(sfrac[i00][0], sfrac[i10][0], sfrac[i01][0], sfrac[i11][0], tx, ty)
        sy_frac = bilerp(sfrac[i00][1], sfrac[i10][1], sfrac[i01][1], sfrac[i11][1], tx, ty)

        sx = int(max(0, min(self.screen_w  - 1, sx_frac * self.screen_w)))
        sy = int(max(0, min(self.screen_h - 1, sy_frac * self.screen_h)))
        return sx, sy


# ── Calibration session UI ────────────────────────────────────────────────────

class CalibrationSession:
    """
    Runs the interactive 9-point calibration session.

    Usage:
        session = CalibrationSession(calib, screen_w, screen_h)
        while True:
            frame = camera.read()
            iris_pt = get_iris(frame)
            done, frame = session.update(frame, iris_pt)
            cv2.imshow(..., frame)
            if cv2.waitKey(1) == ord(' '):
                session.confirm_point()
            if done:
                break
    """

    def __init__(self, calib: GazeCalibration, screen_w: int, screen_h: int):
        self.calib = calib
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.current_idx = 0
        self.samples: List[Tuple[float, float]] = []
        self.collected: List[Tuple[float, float, float, float]] = []
        self.done = False
        self._countdown = 0   # frames left in auto-collect countdown
        self._dot_radius = 20
        self._collecting = False

    @property
    def current_screen_pos(self) -> Tuple[int, int]:
        fx, fy = CALIB_POINTS_FRAC[self.current_idx]
        return int(fx * self.screen_w), int(fy * self.screen_h)

    def confirm_point(self):
        """User pressed SPACE — start collecting samples for this point."""
        if self.done or self._collecting:
            return
        self.samples = []
        self._collecting = True
        self._countdown = SAMPLES_PER_POINT

    def update(self, frame, iris_pt: Optional[Tuple[float, float]]):
        """
        Feed latest camera frame and iris position.
        Returns (done, annotated_frame).
        """
        import cv2
        h, w = frame.shape[:2]

        if self._collecting and iris_pt is not None:
            self.samples.append(iris_pt)
            self._countdown -= 1
            if self._countdown <= 0:
                self._finish_point()

        # ── Draw overlay ──────────────────────────────────────────────────────
        overlay = frame.copy()

        # Semi-transparent black background
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        if not self.done:
            fx, fy = CALIB_POINTS_FRAC[self.current_idx]
            # Map screen fraction → frame pixel (frame may be smaller than screen)
            px = int(fx * w)
            py = int(fy * h)

            # Pulse radius while collecting
            r = self._dot_radius
            color = (0, 255, 0) if self._collecting else (0, 80, 255)
            if self._collecting:
                filled = int(self._dot_radius * (1 - self._countdown / SAMPLES_PER_POINT))
                cv2.circle(frame, (px, py), self._dot_radius, color, 2)
                cv2.circle(frame, (px, py), max(1, filled), color, -1)
            else:
                cv2.circle(frame, (px, py), r, color, -1)
                cv2.circle(frame, (px, py), r + 4, (255, 255, 255), 2)

            # Progress dots at bottom
            for i, (bfx, bfy) in enumerate(CALIB_POINTS_FRAC):
                bx = int(bfx * w)
                by = int(bfy * h)
                bc = (0, 200, 0) if i < self.current_idx else (60, 60, 60)
                cv2.circle(frame, (bx, by), 5, bc, -1)

            n = self.current_idx + 1
            status = "Recording..." if self._collecting else "Look at the dot, then press SPACE"
            cv2.putText(frame, f"Calibration  {n}/9  —  {status}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Calibration complete!",
                (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return self.done, frame

    def _finish_point(self):
        if not self.samples:
            self._collecting = False
            return
        ix = sum(s[0] for s in self.samples) / len(self.samples)
        iy = sum(s[1] for s in self.samples) / len(self.samples)
        fx, fy = CALIB_POINTS_FRAC[self.current_idx]
        self.collected.append((ix, iy, fx, fy))
        print(f"[GazeCalib] Point {self.current_idx+1}/9: iris=({ix:.4f},{iy:.4f}) → screen=({fx:.1f},{fy:.1f})")

        self._collecting = False
        self.samples = []
        self.current_idx += 1

        if self.current_idx >= 9:
            self._complete()

    def _complete(self):
        # Centre = average of all iris readings (or use the centre point directly)
        centre_iris = self.collected[4]   # index 4 = screen centre (0.5, 0.5)
        self.calib.cx = centre_iris[0]
        self.calib.cy = centre_iris[1]
        self.calib.points = [list(p) for p in self.collected]
        self.calib.calibrated = True
        self.calib.save()
        self.done = True
        print("[GazeCalib] 9-point calibration complete and saved.")