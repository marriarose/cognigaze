import time
import math
from typing import Optional, Tuple
from collections import deque

class DwellDetector:
    """
    Detects when the gaze stays within a stable, small region (low variance) for a defined dwell time.
    """
    def __init__(self, dwell_time: float = 1.0, radius_px: int = 20, variance_window: int = 15):
        """
        Args:
            dwell_time: Time in seconds required to trigger a click.
            radius_px: Maximum allowed spatial variance (bounding box diagonal) to maintain dwell.
            variance_window: Frame count to calculate stability.
        """
        self.dwell_time = dwell_time
        self.radius_px = radius_px
        self.variance_window = variance_window
        
        self.position_history = deque(maxlen=variance_window)
        self.start_time: Optional[float] = None
        self.is_dwelling = False

    def update(self, current_x: int, current_y: int) -> bool:
        """
        Update the dwell detector with the current gaze position.
        
        Args:
            current_x: X coordinate of gaze in pixels
            current_y: Y coordinate of gaze in pixels
            
        Returns:
            True if a click should be triggered due to dwell, False otherwise
        """
        now = time.time()
        self.position_history.append((current_x, current_y))
        
        # Need enough history to compute variance/stability robustly
        if len(self.position_history) < self.variance_window:
            return False
            
        # Calculate bounding box of the recent history window
        min_x = min(p[0] for p in self.position_history)
        max_x = max(p[0] for p in self.position_history)
        min_y = min(p[1] for p in self.position_history)
        max_y = max(p[1] for p in self.position_history)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Measure maximum spatial spread (diagonal of the bounding box)
        dist_spread = math.hypot(width, height)
        
        # If the spatial spread is small enough, the gaze is considered stable
        if dist_spread <= self.radius_px:
            if not self.is_dwelling:
                # Target acquired, start the dwell timer
                self.is_dwelling = True
                self.start_time = now
                return False
            else:
                # We are actively dwelling in a stable region
                start = self.start_time
                if start is not None and now - start >= self.dwell_time:
                    # Timer fulfilled! Trigger click & clear stability to prevent double clicks
                    self.is_dwelling = False
                    self.start_time = None
                    self.position_history.clear()
                    return True
        else:
            # The spatial spread exceeded our threshold, meaning the eye darted away / destabilized
            self.is_dwelling = False
            self.start_time = None
            
        return False
