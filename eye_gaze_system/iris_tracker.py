"""Iris tracking module using MediaPipe Iris landmarks with fallback pupil localization."""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from .face_landmarks import FaceLandmarkDetector


class IrisTracker:
    """Tracks iris position from MediaPipe Iris landmarks with fallback to image processing."""
    
    def __init__(
        self, 
        face_detector: FaceLandmarkDetector, 
        use_both_eyes: bool = False,
        enable_fallback: bool = True
    ):
        """
        Initialize the iris tracker.
        
        Args:
            face_detector: FaceLandmarkDetector instance
            use_both_eyes: If True, average both eyes; if False, use left eye only
            enable_fallback: Enable fallback pupil localization when MediaPipe fails
        """
        self.face_detector = face_detector
        self.use_both_eyes = use_both_eyes
        self.enable_fallback = enable_fallback
    
    def _localize_pupil_fallback(
        self,
        eye_roi: np.ndarray,
        roi_x: int,
        roi_y: int
    ) -> Optional[Tuple[int, int]]:
        """
        Robust pupil localization using image processing (fallback method).
        
        This method is used when MediaPipe confidence drops or detection fails.
        
        Args:
            eye_roi: Eye region of interest (BGR image)
            roi_x: X coordinate of ROI in full frame
            roi_y: Y coordinate of ROI in full frame
            
        Returns:
            Tuple of (x, y) pixel coordinates of pupil center in full frame, or None
        """
        if eye_roi is None or eye_roi.size == 0:
            return None
        
        # Step 1: Convert to grayscale
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi.copy()
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Apply adaptive threshold
        # Using adaptive threshold to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Step 4: Morphological open to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Step 5: Compute centroid using cv2.moments
        moments = cv2.moments(opened)
        
        if moments["m00"] != 0:
            # Calculate centroid
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Convert to full frame coordinates
            pupil_x = roi_x + cx
            pupil_y = roi_y + cy
            
            return (pupil_x, pupil_y)
        
        # Step 6: Fallback to Hough Circle Transform if moments fail
        return self._localize_pupil_hough(eye_roi, roi_x, roi_y)
    
    def _localize_pupil_hough(
        self,
        eye_roi: np.ndarray,
        roi_x: int,
        roi_y: int
    ) -> Optional[Tuple[int, int]]:
        """
        Backup method using Hough Circle Transform.
        
        Args:
            eye_roi: Eye region of interest (BGR image)
            roi_x: X coordinate of ROI in full frame
            roi_y: Y coordinate of ROI in full frame
            
        Returns:
            Tuple of (x, y) pixel coordinates of pupil center in full frame, or None
        """
        if eye_roi is None or eye_roi.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Estimate pupil radius (typically 1/4 to 1/3 of eye width)
        roi_height, roi_width = blurred.shape[:2]
        min_radius = max(3, int(min(roi_width, roi_height) * 0.1))
        max_radius = int(min(roi_width, roi_height) * 0.4)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None and len(circles) > 0:
            # Get the first (most confident) circle
            circle = circles[0][0]
            cx = int(circle[0])
            cy = int(circle[1])
            
            # Convert to full frame coordinates
            pupil_x = roi_x + cx
            pupil_y = roi_y + cy
            
            return (pupil_x, pupil_y)
        
        return None
    
    def track_with_fallback(
        self,
        landmarks: Optional[List],
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        eye: str = 'left'
    ) -> Optional[Tuple[float, float]]:
        """
        Track iris with fallback to image processing if MediaPipe fails.
        
        Args:
            landmarks: List of facial landmark objects (can be None)
            frame: Full frame (BGR format)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            eye: Which eye ('left' or 'right')
            
        Returns:
            Tuple of normalized (x, y) coordinates of iris center, or None
        """
        # Try MediaPipe first
        if landmarks is not None:
            center_3d = self.face_detector.get_iris_center_3d(landmarks, eye)
            if center_3d is not None:
                return (center_3d[0], center_3d[1])
        
        # Fallback to image processing if MediaPipe failed
        if not self.enable_fallback:
            return None
        
        # Get eye ROI
        if landmarks is not None:
            roi = self.face_detector.get_eye_roi(landmarks, frame_width, frame_height, eye)
        else:
            # If no landmarks, can't get ROI - return None
            return None
        
        if roi is None:
            return None
        
        roi_x, roi_y, roi_w, roi_h = roi
        
        # Extract ROI from frame
        eye_roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Localize pupil using image processing
        pupil_center = self._localize_pupil_fallback(eye_roi, roi_x, roi_y)
        
        if pupil_center is None:
            return None
        
        # Convert to normalized coordinates
        normalized_x = pupil_center[0] / frame_width
        normalized_y = pupil_center[1] / frame_height
        
        return (normalized_x, normalized_y)
    
    def track(
        self, 
        landmarks: Optional[List],
        frame_width: int = 640,
        frame_height: int = 480,
        frame: Optional[np.ndarray] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Track the iris center position (normalized coordinates).
        
        Args:
            landmarks: List of facial landmark objects (can be None for fallback)
            frame_width: Frame width (for 2D coordinate conversion if needed)
            frame_height: Frame height (for 2D coordinate conversion if needed)
            frame: Full frame (BGR format) - required for fallback
            
        Returns:
            Tuple of normalized (x, y) coordinates of iris center, or None
        """
        # Try MediaPipe first
        if landmarks is not None:
            center_3d = self.face_detector.get_iris_center_3d(landmarks, eye='left')
            
            if center_3d is not None:
                if self.use_both_eyes:
                    # Average both eyes
                    right_center_3d = self.face_detector.get_iris_center_3d(landmarks, eye='right')
                    if right_center_3d is not None:
                        # Average x and y coordinates
                        x = (center_3d[0] + right_center_3d[0]) / 2.0
                        y = (center_3d[1] + right_center_3d[1]) / 2.0
                        return (x, y)
                
                # Return left eye center (x, y) normalized coordinates
                return (center_3d[0], center_3d[1])
        
        # Fallback to image processing if MediaPipe failed and frame is provided
        if self.enable_fallback and frame is not None:
            return self.track_with_fallback(landmarks, frame, frame_width, frame_height, 'left')
        
        return None
    
    def track_2d(
        self,
        landmarks: Optional[List],
        frame_width: int,
        frame_height: int,
        eye: str = 'left',
        frame: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Track the iris center in 2D pixel coordinates with fallback.
        
        Args:
            landmarks: List of facial landmark objects (can be None)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            eye: Which eye ('left' or 'right')
            frame: Full frame (BGR format) - required for fallback
            
        Returns:
            Tuple of (x, y) pixel coordinates of iris center, or None
        """
        # Try MediaPipe first
        if landmarks is not None:
            center_2d = self.face_detector.get_iris_center_2d(landmarks, frame_width, frame_height, eye)
            if center_2d is not None:
                return center_2d
        
        # Fallback to image processing
        if not self.enable_fallback or frame is None:
            return None
        
        # Get eye ROI
        if landmarks is not None:
            roi = self.face_detector.get_eye_roi(landmarks, frame_width, frame_height, eye)
        else:
            return None
        
        if roi is None:
            return None
        
        roi_x, roi_y, roi_w, roi_h = roi
        
        # Extract ROI from frame
        eye_roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Localize pupil using image processing
        return self._localize_pupil_fallback(eye_roi, roi_x, roi_y)
    
    def get_iris_landmarks_3d(
        self, 
        landmarks: List,
        eye: str = 'left'
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Get all iris landmark positions in 3D normalized coordinates.
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye ('left' or 'right')
            
        Returns:
            List of (x, y, z) normalized coordinates for all iris landmarks, or None
        """
        if eye == 'left':
            return self.face_detector.get_left_iris_landmarks(landmarks)
        elif eye == 'right':
            return self.face_detector.get_right_iris_landmarks(landmarks)
        else:
            return None
    
    def get_iris_data(
        self,
        landmarks: List,
        frame_width: int,
        frame_height: int,
        eye: str = 'left'
    ) -> Optional[Dict]:
        """
        Get complete iris data including 2D center, 3D coordinates, and ROI.
        
        Args:
            landmarks: List of facial landmark objects
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            eye: Which eye ('left' or 'right')
            
        Returns:
            Dictionary with iris data, or None
        """
        return self.face_detector.get_iris_data(landmarks, frame_width, frame_height, eye)
