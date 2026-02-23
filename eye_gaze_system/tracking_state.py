"""Tracking state management for handling edge cases."""

from typing import Optional, Tuple
from collections import deque
import time
import numpy as np


class TrackingState:
    """
    Manages tracking state and handles edge cases:
    - Head movement
    - Partial occlusion
    - Frame drops
    - Data loss
    - Low confidence
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        freeze_threshold: float = 0.3,
        max_freeze_frames: int = 10,
        stability_window: int = 5,
        max_frame_drop: int = 30,
    ):
        """
        Initialize tracking state manager.
        
        Args:
            confidence_threshold: Minimum confidence to consider tracking valid
            freeze_threshold: Confidence below which cursor should freeze
            max_freeze_frames: Maximum frames to freeze before attempting recovery
            stability_window: Window size for stability checking
            max_frame_drop: Frames dropped (100ms+) before freezing cursor (default 30)
        """
        self.confidence_threshold = confidence_threshold
        self.freeze_threshold = freeze_threshold
        self.max_freeze_frames = max_freeze_frames
        self.stability_window = stability_window
        self.max_frame_drop = max_frame_drop
        
        # State tracking
        self.current_confidence = 1.0
        self.freeze_frames = 0
        self.last_valid_position: Optional[Tuple[float, float]] = None
        self.is_frozen = False
        
        # History for stability checking
        self.position_history = deque(maxlen=stability_window)
        self.confidence_history = deque(maxlen=stability_window)
        
        # Frame drop detection
        self.last_frame_time = time.time()
        self.frame_drop_count = 0
        
        # Head movement detection
        self.position_variance = 0.0
        self.movement_threshold = 0.1  # Normalized coordinate variance
        
    def update_confidence(self, confidence: float):
        """
        Update tracking confidence.
        
        Args:
            confidence: Current confidence value [0, 1]
        """
        self.current_confidence = confidence
        self.confidence_history.append(confidence)
        
        # Check if confidence dropped below threshold
        if confidence < self.freeze_threshold:
            self.freeze_frames += 1
            self.is_frozen = True
        elif confidence >= self.confidence_threshold:
            self.freeze_frames = 0
            self.is_frozen = False
    
    def update_position(self, position: Optional[Tuple[float, float]]):
        """
        Update tracking position and check for stability.
        
        Args:
            position: Current position (x, y) or None if lost
        """
        if position is not None:
            self.last_valid_position = position
            self.position_history.append(position)
            
            # Check for head movement (high variance)
            if len(self.position_history) >= self.stability_window:
                positions = np.array(list(self.position_history))
                self.position_variance = np.var(positions, axis=0).mean()
        else:
            # Data loss - position is None
            self.position_history.append(None)
    
    def should_freeze_cursor(self) -> bool:
        """
        Determine if cursor should be frozen.
        
        Returns:
            True if cursor should be frozen, False otherwise
        """
        # Freeze if confidence is too low
        if self.current_confidence < self.freeze_threshold:
            return True
        
        # Freeze if too many frames dropped
        if self.frame_drop_count > self.max_frame_drop:
            return True
        
        # Don't freeze if we've been frozen too long (attempt recovery)
        if self.freeze_frames > self.max_freeze_frames:
            self.freeze_frames = 0
            return False
        
        return self.is_frozen
    
    def should_use_fallback(self) -> bool:
        """
        Determine if fallback pupil detection should be used.
        
        Returns:
            True if fallback should be attempted
        """
        # Use fallback if confidence is low
        if self.current_confidence < self.confidence_threshold:
            return True
        
        # Use fallback if tracking is frozen
        if self.is_frozen:
            return True
        
        return False
    
    def get_frozen_position(self) -> Optional[Tuple[float, float]]:
        """
        Get the last valid position to use when frozen.
        
        Returns:
            Last valid position, or None if never had valid position
        """
        return self.last_valid_position
    
    def check_frame_drop(self) -> bool:
        """
        Check if frame was dropped (too much time since last frame).
        
        Returns:
            True if frame drop detected
        """
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        
        # If more than 100ms since last frame, consider it dropped
        if time_delta > 0.1:
            self.frame_drop_count += 1
            self.last_frame_time = current_time
            return True
        else:
            self.frame_drop_count = 0
            self.last_frame_time = current_time
            return False
    
    def is_head_moving(self) -> bool:
        """
        Check if head is moving significantly.
        
        Returns:
            True if head movement detected
        """
        return self.position_variance > self.movement_threshold
    
    def is_partially_occluded(self) -> bool:
        """
        Check if face is partially occluded (low confidence but not zero).
        
        Returns:
            True if partial occlusion detected
        """
        # Partial occlusion: confidence between freeze and threshold
        return (self.freeze_threshold <= self.current_confidence < self.confidence_threshold)
    
    def reset(self):
        """Reset tracking state."""
        self.current_confidence = 1.0
        self.freeze_frames = 0
        self.is_frozen = False
        self.position_history.clear()
        self.confidence_history.clear()
        self.frame_drop_count = 0
        self.position_variance = 0.0
        self.last_frame_time = time.time()
