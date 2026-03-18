import math

class GazeProcessor:
    """
    Lightweight processor for 1:1 screen tracking and responsive targeting.
    Designed to process normalized coordinates [0.0, 1.0].
    """
    def __init__(self):
        # Responsiveness parameters
        self.alpha = 0.70              # Light EMA (0.6 - 0.8) for responsiveness
        
        self.prev_smooth = None

    def map_and_smooth(self, raw_x, raw_y):
        """
        raw_x, raw_y: Normalized gaze coordinates [0, 1].
        """
        # Ensure purely bounded coordinates
        mx = self._clamp(raw_x, 0.0, 1.0)
        my = self._clamp(raw_y, 0.0, 1.0)
        
        # Initialize
        if self.prev_smooth is None:
            self.prev_smooth = (mx, my)
            return mx, my
            
        # Light Exponential Moving Average (EMA)
        smooth_x = self.prev_smooth[0] * (1.0 - self.alpha) + mx * self.alpha
        smooth_y = self.prev_smooth[1] * (1.0 - self.alpha) + my * self.alpha
        
        # Hard constraint to keep output perfectly within screen bounds
        smooth_x = self._clamp(smooth_x, 0.0, 1.0)
        smooth_y = self._clamp(smooth_y, 0.0, 1.0)
        
        self.prev_smooth = (smooth_x, smooth_y)
        
        return smooth_x, smooth_y
        
    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))
