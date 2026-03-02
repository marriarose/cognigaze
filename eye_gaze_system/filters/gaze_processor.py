import numpy as np
import math
from .kalman_filter import KalmanFilter

class GazeProcessor:
    """
    Advanced processor for screen edge mapping correction, noise reduction, and precision targeting.
    Designed to process normalized coordinates [0.0, 1.0].
    """
    def __init__(self):
        # Recommended Parameters
        self.padding = 0.12            # 12% artificial padding
        self.amp_factor = 2.0          # Cubic amplification
        self.dead_zone = 0.003         # Normalized spatial threshold
        self.alpha_min = 0.1           # High smoothing for precision
        self.alpha_max = 0.6           # Low smoothing for speed
        self.vel_max = 0.05            # Velocity saturation limit
        
        self.prev_raw = None
        self.prev_smooth = None
        self._stationary_frames = 0
        
        # Instantiate the existing Kalman filter with precision-tuned noise settings
        self.kalman = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        
    def map_and_smooth(self, raw_x, raw_y):
        """
        raw_x, raw_y: Normalized gaze coordinates [0, 1].
        """
        
        # 1. Edge Compensation (Padding)
        px = self._clamp((raw_x - self.padding) / (1.0 - 2 * self.padding), 0.0, 1.0)
        py = self._clamp((raw_y - self.padding) / (1.0 - 2 * self.padding), 0.0, 1.0)
        
        # 2. Cubic Edge Mapping
        mx = self._cubic_amplify(px, self.amp_factor)
        my = self._cubic_amplify(py, self.amp_factor)
        
        # Initialize
        if self.prev_raw is None:
            self.prev_raw = (mx, my)
            self.prev_smooth = (mx, my)
            self._stationary_frames = 0
            # Send initial measurement to initialize Kalman filter
            self.kalman.update((mx, my))
            return mx, my
            
        # 3. Micro-jitter suppression & Freeze logic
        dx = mx - self.prev_raw[0]
        dy = my - self.prev_raw[1]
        dist = math.hypot(dx, dy)
        
        # Dead-zone (velocity zeroing)
        if dist < self.dead_zone:
            mx, my = self.prev_raw
            self._stationary_frames += 1
            dist = 0.0
        else:
            self._stationary_frames = 0
            
        # 4. Kalman Filtering
        kx, ky = self.kalman.filter((mx, my))
        
        # 5. Dynamic EMA (Exponential Moving Average) Target
        # If stationary for extended time (>N frames), hard-freeze the cursor to stop minor drift
        if self._stationary_frames > 5:
            smooth_x, smooth_y = self.prev_smooth
            # Keep Kalman state updated with frozen coords so it doesn't build up latent error
            self.kalman.state[0] = smooth_x
            self.kalman.state[1] = smooth_y
            self.kalman.state[2] = 0.0 # velocity x
            self.kalman.state[3] = 0.0 # velocity y
        else:
            # Velocity-based Alpha
            vel_ratio = self._clamp(dist / self.vel_max, 0.0, 1.0)
            alpha = self.alpha_min + vel_ratio * (self.alpha_max - self.alpha_min)
            
            smooth_x = alpha * kx + (1 - alpha) * self.prev_smooth[0]
            smooth_y = alpha * ky + (1 - alpha) * self.prev_smooth[1]
            
        # Hard constraint AFTER smoothing to prevent NaN or screen exit
        smooth_x = self._clamp(smooth_x, 0.01, 0.99) # Keep 1% margin from absolute edges
        smooth_y = self._clamp(smooth_y, 0.01, 0.99)
        
        self.prev_raw = (mx, my)
        self.prev_smooth = (smooth_x, smooth_y)
        
        return smooth_x, smooth_y
        
    def _cubic_amplify(self, t, a):
        tc = t - 0.5
        val = a * (tc ** 3) + (1 - 0.25 * a) * tc + 0.5
        return self._clamp(val, 0.0, 1.0)

    def _clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))
