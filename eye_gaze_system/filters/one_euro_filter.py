"""One Euro filter for smoothing gaze coordinates."""

from typing import Tuple, Optional
import math


class OneEuroFilter:
    """One Euro filter for real-time smoothing with adaptive cutoff."""
    
    def __init__(
        self, 
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0
    ):
        """
        Initialize the One Euro filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency
            beta: Speed coefficient (higher = more responsive)
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = None
        self.dy_prev = None
        
        self.initialized = False
    
    def _alpha(self, cutoff: float, dt: float) -> float:
        """Calculate alpha parameter."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def update(
        self, 
        measurement: Tuple[float, float],
        dt: float = 0.033  # Default 30 FPS
    ) -> Tuple[float, float]:
        """
        Update the filter with a new measurement.
        
        Args:
            measurement: Tuple of (x, y) coordinates
            dt: Time delta since last update (in seconds)
            
        Returns:
            Filtered (x, y) coordinates
        """
        x, y = measurement
        
        if not self.initialized:
            self.x_prev = x
            self.y_prev = y
            self.dx_prev = 0.0
            self.dy_prev = 0.0
            self.initialized = True
            return measurement
        
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        dy = (y - self.y_prev) / dt
        
        # Smooth derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        dy_hat = alpha_d * dy + (1 - alpha_d) * self.dy_prev
        
        # Adaptive cutoff based on derivative
        cutoff = self.min_cutoff + self.beta * math.sqrt(dx_hat**2 + dy_hat**2)
        
        # Smooth position
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        y_hat = alpha * y + (1 - alpha) * self.y_prev
        
        # Update state
        self.x_prev = x_hat
        self.y_prev = y_hat
        self.dx_prev = dx_hat
        self.dy_prev = dy_hat
        
        return (x_hat, y_hat)
    
    def set_min_cutoff(self, min_cutoff: float):
        """
        Update the minimum cutoff frequency.
        
        Args:
            min_cutoff: New minimum cutoff frequency
        """
        self.min_cutoff = min_cutoff
    
    def set_beta(self, beta: float):
        """
        Update the beta parameter (speed coefficient).
        
        Args:
            beta: New beta value (higher = more responsive)
        """
        self.beta = beta
    
    def set_d_cutoff(self, d_cutoff: float):
        """
        Update the derivative cutoff frequency.
        
        Args:
            d_cutoff: New derivative cutoff frequency
        """
        self.d_cutoff = d_cutoff
    
    def reset(self):
        """Reset the filter state."""
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = None
        self.dy_prev = None
        self.initialized = False
