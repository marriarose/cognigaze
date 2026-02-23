"""Blink detection module using EAR (Eye Aspect Ratio)."""

import numpy as np
from typing import Optional, Tuple, List
import time
from .face_landmarks import FaceLandmarkDetector


class BlinkDetector:
    """
    Detects eye blinks using EAR (Eye Aspect Ratio).
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
    
    If EAR < threshold for X consecutive frames:
    - Short blink: left click
    - Long blink: right click
    """
    
    # Default EAR threshold (typically 0.2-0.3 for closed eye)
    DEFAULT_EAR_THRESHOLD = 0.25
    
    def __init__(
        self, 
        face_detector: FaceLandmarkDetector,
        ear_threshold: float = 0.25,
        consecutive_frames: int = 3,
        long_blink_frames: int = 10,
        debounce_time: float = 0.3
    ):
        """
        Initialize the EAR-based blink detector.
        
        Args:
            face_detector: FaceLandmarkDetector instance
            ear_threshold: EAR threshold below which eye is considered closed
            consecutive_frames: Number of consecutive frames with low EAR to trigger blink
            long_blink_frames: Number of consecutive frames for long blink (right click)
            debounce_time: Minimum time between clicks (seconds) to prevent spamming
        """
        self.face_detector = face_detector
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.long_blink_frames = long_blink_frames
        self.debounce_time = debounce_time
        
        # State tracking
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.last_click_time = 0.0
        self.blink_state = 'open'  # 'open', 'blinking', 'long_blink'
    
    def _compute_distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float]
    ) -> float:
        """
        Compute Euclidean distance between two 2D points.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_ear(
        self,
        landmarks: List,
        eye: str = 'left'
    ) -> Optional[float]:
        """
        Compute Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye ('left' or 'right')
            
        Returns:
            EAR value, or None if calculation fails
        """
        ear_points = self.face_detector.get_ear_landmarks(landmarks, eye)
        
        if ear_points is None or len(ear_points) != 6:
            return None
        
        # Extract points: [p1, p2, p3, p4, p5, p6]
        p1, p2, p3, p4, p5, p6 = ear_points
        
        # Compute distances
        # Vertical distances
        d1 = self._compute_distance(p2, p6)  # ||p2-p6||
        d2 = self._compute_distance(p3, p5)  # ||p3-p5||
        
        # Horizontal distance
        d3 = self._compute_distance(p1, p4)  # ||p1-p4||
        
        # Avoid division by zero
        if d3 < 1e-6:
            return None
        
        # Compute EAR
        ear = (d1 + d2) / (2.0 * d3)
        
        return ear
    
    def detect_blink(
        self, 
        landmarks: Optional[List],
        eye: str = 'left'
    ) -> str:
        """
        Detect blink state using EAR.
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye to check ('left' or 'right')
            
        Returns:
            Blink state: 'none', 'short_blink', 'long_blink'
        """
        if landmarks is None:
            return 'none'
        
        # Compute EAR
        ear = self.compute_ear(landmarks, eye)
        
        if ear is None:
            return 'none'
        
        # Check if eye is closed (EAR below threshold)
        is_closed = ear < self.ear_threshold
        
        # Update frame counter
        if eye == 'left':
            if is_closed:
                self.left_eye_closed_frames += 1
            else:
                self.left_eye_closed_frames = 0
            closed_frames = self.left_eye_closed_frames
        else:
            if is_closed:
                self.right_eye_closed_frames += 1
            else:
                self.right_eye_closed_frames = 0
            closed_frames = self.right_eye_closed_frames
        
        # Determine blink state
        if closed_frames >= self.long_blink_frames:
            return 'long_blink'
        elif closed_frames >= self.consecutive_frames:
            return 'short_blink'
        else:
            return 'none'
    
    def should_trigger_click(
        self,
        landmarks: Optional[List],
        eye: str = 'left'
    ) -> Tuple[bool, str]:
        """
        Check if a click should be triggered based on blink detection.
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye to check ('left' or 'right')
            
        Returns:
            Tuple of (should_click, click_type) where:
                - should_click: True if click should be triggered
                - click_type: 'left' for left click, 'right' for right click, 'none' for no click
        """
        # Check debounce timer
        current_time = time.time()
        if current_time - self.last_click_time < self.debounce_time:
            return (False, 'none')
        
        # Detect blink state
        blink_state = self.detect_blink(landmarks, eye)
        
        if blink_state == 'long_blink':
            # Long blink: right click
            self.last_click_time = current_time
            # Reset frame counter to prevent multiple clicks
            if eye == 'left':
                self.left_eye_closed_frames = 0
            else:
                self.right_eye_closed_frames = 0
            return (True, 'right')
        elif blink_state == 'short_blink':
            # Short blink: left click
            self.last_click_time = current_time
            # Reset frame counter to prevent multiple clicks
            if eye == 'left':
                self.left_eye_closed_frames = 0
            else:
                self.right_eye_closed_frames = 0
            return (True, 'left')
        else:
            return (False, 'none')
    
    def get_ear_value(
        self,
        landmarks: Optional[List],
        eye: str = 'left'
    ) -> Optional[float]:
        """
        Get current EAR value without triggering clicks.
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye to check ('left' or 'right')
            
        Returns:
            EAR value, or None if calculation fails
        """
        if landmarks is None:
            return None
        
        return self.compute_ear(landmarks, eye)
    
    def set_ear_threshold(self, threshold: float):
        """
        Update the EAR threshold.
        
        Args:
            threshold: New EAR threshold value
        """
        self.ear_threshold = threshold
    
    def set_consecutive_frames(self, frames: int):
        """
        Update the number of consecutive frames required for blink detection.
        
        Args:
            frames: New consecutive frames value
        """
        self.consecutive_frames = frames
    
    def set_long_blink_frames(self, frames: int):
        """
        Update the number of frames required for long blink (right click).
        
        Args:
            frames: New long blink frames value
        """
        self.long_blink_frames = frames
    
    def set_debounce_time(self, debounce_time: float):
        """
        Update the debounce time.
        
        Args:
            debounce_time: New debounce time in seconds
        """
        self.debounce_time = debounce_time
    
    def reset(self):
        """Reset the detector state."""
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.last_click_time = 0.0
        self.blink_state = 'open'
