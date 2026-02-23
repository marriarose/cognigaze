"""Weighted average filter for smoothing gaze coordinates."""

from typing import Tuple, Optional
from collections import deque
import numpy as np


class WeightedAverageFilter:
    """
    Weighted average filter using exponential weighting.
    
    Recent measurements get higher weight than older ones.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        alpha: float = 0.7
    ):
        """
        Initialize the weighted average filter.
        
        Args:
            window_size: Size of the sliding window
            alpha: Weighting factor (0-1). Higher = more weight on recent values.
                   alpha=0.7 means recent values have 70% weight, older values decay exponentially.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
    
    def filter(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply weighted average filtering.
        
        Args:
            measurement: Tuple of (x, y) coordinates
            
        Returns:
            Filtered (x, y) coordinates
        """
        x, y = measurement
        
        # Add to history
        self.history.append(measurement)
        
        if len(self.history) == 1:
            # First measurement, return as-is
            return measurement
        
        # Compute weighted average
        # More recent measurements get higher weight
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        # Iterate from most recent to oldest
        for i, (hist_x, hist_y) in enumerate(reversed(self.history)):
            # Weight decreases exponentially for older values
            weight = self.alpha ** i
            total_weight += weight
            weighted_x += hist_x * weight
            weighted_y += hist_y * weight
        
        # Normalize
        if total_weight > 0:
            weighted_x /= total_weight
            weighted_y /= total_weight
        
        return (weighted_x, weighted_y)
    
    def set_alpha(self, alpha: float):
        """
        Update the weighting factor.
        
        Args:
            alpha: New alpha value (0-1)
        """
        self.alpha = max(0.0, min(1.0, alpha))
    
    def reset(self):
        """Reset the filter state."""
        self.history.clear()
