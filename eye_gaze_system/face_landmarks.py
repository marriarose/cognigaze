"""Face landmark detection using MediaPipe Face Mesh with Iris tracking."""

import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2


class FaceLandmarkDetector:
    """Detects facial landmarks using MediaPipe Face Mesh with refined iris landmarks."""

    # ─────────────────────────────────────────────────────────────────────────
    # MediaPipe FaceMesh refined landmark layout (478 total, 0-indexed)
    #
    # "Left" / "Right" = person's anatomical left / right.
    #
    # IRIS (only with refine_landmarks=True):
    #   Person's LEFT  eye → LEFT_IRIS_CENTER  = 468  (true pupil centre)
    #                         boundary = 469, 470, 471, 472
    #   Person's RIGHT eye → RIGHT_IRIS_CENTER = 473  (true pupil centre)
    #                         boundary = 474, 475, 476, 477
    #
    # EYE CONTOUR corners / lids:
    #   Person's RIGHT eye → outer corner=33, inner corner=133, top=159, bottom=145
    #   Person's LEFT  eye → outer corner=263, inner corner=362, top=386, bottom=374
    #
    # GAZE EYE: we use person's RIGHT eye + RIGHT iris (473) together.
    # In mirror-mode (flipped) preview this eye appears on the right of the
    # window, which is natural for the user.
    # ─────────────────────────────────────────────────────────────────────────

    # Iris centres
    LEFT_IRIS_CENTER  = 468
    LEFT_IRIS_START   = 469
    LEFT_IRIS_END     = 473   # exclusive; range(469,473) = [469,470,471,472]

    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_START  = 474
    RIGHT_IRIS_END    = 478   # exclusive; range(474,478) = [474,475,476,477]

    # Gaze eye = person's RIGHT eye + RIGHT iris — must match each other
    _GAZE_EYE_OUTER_CORNER = 33    # temporal corner, person's right eye
    _GAZE_EYE_INNER_CORNER = 133   # nasal corner,    person's right eye
    _GAZE_EYE_TOP_LID      = 159   # upper lid,       person's right eye
    _GAZE_EYE_BOTTOM_LID   = 145   # lower lid,       person's right eye
    _GAZE_IRIS_CENTER      = 473   # RIGHT iris centre — SAME eye as above

    # Eye contour rings (for ROI / socket centre)
    LEFT_EYE_CONTOUR  = [33, 7, 163, 144, 145, 153, 154, 155,
                          133, 173, 157, 158, 159, 160, 161, 246]   # right eye
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                          263, 466, 388, 387, 386, 385, 384, 398]   # left eye

    # Blink lid pairs  (API keeps "left"/"right" names for backward compat)
    LEFT_EYE_TOP     = 159   # person's right eye upper lid
    LEFT_EYE_BOTTOM  = 145   # person's right eye lower lid
    RIGHT_EYE_TOP    = 386   # person's left  eye upper lid
    RIGHT_EYE_BOTTOM = 374   # person's left  eye lower lid

    # EAR sets: [outer-corner, top1, top2, inner-corner, bottom1, bottom2]
    LEFT_EYE_EAR  = [33,  160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, refine_landmarks: bool = True):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=refine_landmarks,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.refine_landmarks = refine_landmarks

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, rgb_frame: np.ndarray) -> Tuple[Optional[List], float]:
        output = self.face_mesh.process(rgb_frame)
        if output.multi_face_landmarks and len(output.multi_face_landmarks) > 0:
            landmarks = output.multi_face_landmarks[0].landmark
            n = len(landmarks)
            if n >= 478:
                confidence = 1.0
            elif n >= 468:
                confidence = 0.9
            elif n >= 200:
                confidence = 0.7
            else:
                confidence = 0.3
            return landmarks, confidence
        return None, 0.0

    # ── Iris landmark accessors ───────────────────────────────────────────────

    def get_left_iris_landmarks(
        self, landmarks: List
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Person's LEFT eye iris. Index 0 = true centre (landmark 468)."""
        if landmarks is None or len(landmarks) < self.LEFT_IRIS_END:
            return None
        c = landmarks[self.LEFT_IRIS_CENTER]
        result = [(c.x, c.y, c.z)]
        for i in range(self.LEFT_IRIS_START, self.LEFT_IRIS_END):
            lm = landmarks[i]
            result.append((lm.x, lm.y, lm.z))
        return result

    def get_right_iris_landmarks(
        self, landmarks: List
    ) -> Optional[List[Tuple[float, float, float]]]:
        """Person's RIGHT eye iris. Index 0 = true centre (landmark 473)."""
        if landmarks is None or len(landmarks) < self.RIGHT_IRIS_END:
            return None
        c = landmarks[self.RIGHT_IRIS_CENTER]
        result = [(c.x, c.y, c.z)]
        for i in range(self.RIGHT_IRIS_START, self.RIGHT_IRIS_END):
            lm = landmarks[i]
            result.append((lm.x, lm.y, lm.z))
        return result

    def get_iris_center_2d(
        self,
        landmarks: List,
        frame_width: int,
        frame_height: int,
        eye: str = 'left',
    ) -> Optional[Tuple[int, int]]:
        lms = (self.get_left_iris_landmarks(landmarks)
               if eye == 'left'
               else self.get_right_iris_landmarks(landmarks))
        if not lms:
            return None
        return (int(lms[0][0] * frame_width), int(lms[0][1] * frame_height))

    def get_iris_center_3d(
        self,
        landmarks: List,
        eye: str = 'left',
    ) -> Optional[Tuple[float, float, float]]:
        lms = (self.get_left_iris_landmarks(landmarks)
               if eye == 'left'
               else self.get_right_iris_landmarks(landmarks))
        return lms[0] if lms else None

    # ── Core gaze function ────────────────────────────────────────────────────

    def get_left_eye_relative_iris_position(
        self,
        landmarks: List,
    ) -> Optional[Tuple[float, float]]:
        """
        Returns raw RIGHT iris position (landmark 473) as (iris_x, iris_y)
        in MediaPipe normalised coordinates.

        The eye-box relative approach (dividing by eye-socket width) produces
        rel_x ≈ 2.8 because landmark 473 sits at x≈0.50 while the eye-box
        corners (33, 133) span x≈0.34–0.44 — the iris is outside its own box
        in MediaPipe coordinate space.

        Raw coordinates work correctly:
          iris_x decreases → looking left   (cursor left)
          iris_x increases → looking right  (cursor right)
          iris_y decreases → looking up     (cursor up)
          iris_y increases → looking down   (cursor down)
        No axis inversion needed.
        """
        if landmarks is None or len(landmarks) <= self._GAZE_IRIS_CENTER:
            return None
        iris = landmarks[self._GAZE_IRIS_CENTER]  # landmark 473
        return (iris.x, iris.y)

    # ── Eye contour / ROI helpers ─────────────────────────────────────────────

    def get_eye_contour_landmarks(
        self, landmarks: List, eye: str = 'left'
    ) -> Optional[List[Tuple[float, float, float]]]:
        if landmarks is None:
            return None
        indices = self.LEFT_EYE_CONTOUR if eye == 'left' else self.RIGHT_EYE_CONTOUR
        if len(landmarks) < max(indices) + 1:
            return None
        return [(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in indices]

    def get_eye_socket_center(
        self, landmarks: List, eye: str = 'left'
    ) -> Optional[Tuple[float, float, float]]:
        pts = self.get_eye_contour_landmarks(landmarks, eye)
        if not pts:
            return None
        n = len(pts)
        return (sum(p[0] for p in pts) / n,
                sum(p[1] for p in pts) / n,
                sum(p[2] for p in pts) / n)

    def get_eye_roi(
        self,
        landmarks: List,
        frame_width: int,
        frame_height: int,
        eye: str = 'left',
        padding: float = 0.1,
    ) -> Optional[Tuple[int, int, int, int]]:
        pts = self.get_eye_contour_landmarks(landmarks, eye)
        if not pts:
            return None
        xs = [int(p[0] * frame_width)  for p in pts]
        ys = [int(p[1] * frame_height) for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        px, py = int(w * padding), int(h * padding)
        x  = max(0, x_min - px)
        y  = max(0, y_min - py)
        w2 = min(frame_width  - x, w + 2 * px)
        h2 = min(frame_height - y, h + 2 * py)
        return (x, y, w2, h2)

    def get_iris_data(
        self,
        landmarks: List,
        frame_width: int,
        frame_height: int,
        eye: str = 'left',
    ) -> Optional[Dict]:
        c2d  = self.get_iris_center_2d(landmarks, frame_width, frame_height, eye)
        c3d  = self.get_iris_center_3d(landmarks, eye)
        lms  = (self.get_left_iris_landmarks(landmarks)
                if eye == 'left'
                else self.get_right_iris_landmarks(landmarks))
        roi  = self.get_eye_roi(landmarks, frame_width, frame_height, eye)
        if c2d is None or c3d is None or lms is None:
            return None
        return {'center_2d': c2d, 'center_3d': c3d, 'landmarks_3d': lms, 'roi': roi}

    # ── Blink-detection helpers ───────────────────────────────────────────────

    def get_left_eye_landmarks(
        self, landmarks: List
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if landmarks is None or len(landmarks) < max(self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM) + 1:
            return None
        t = landmarks[self.LEFT_EYE_TOP]
        b = landmarks[self.LEFT_EYE_BOTTOM]
        return ((t.x, t.y), (b.x, b.y))

    def get_right_eye_landmarks(
        self, landmarks: List
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if landmarks is None or len(landmarks) < max(self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM) + 1:
            return None
        t = landmarks[self.RIGHT_EYE_TOP]
        b = landmarks[self.RIGHT_EYE_BOTTOM]
        return ((t.x, t.y), (b.x, b.y))

    def get_ear_landmarks(
        self, landmarks: List, eye: str = 'left'
    ) -> Optional[List[Tuple[float, float]]]:
        if landmarks is None:
            return None
        indices = self.LEFT_EYE_EAR if eye == 'left' else self.RIGHT_EYE_EAR
        if len(landmarks) < max(indices) + 1:
            return None
        return [(landmarks[i].x, landmarks[i].y) for i in indices]

    # ── Utility ───────────────────────────────────────────────────────────────

    def frame_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)