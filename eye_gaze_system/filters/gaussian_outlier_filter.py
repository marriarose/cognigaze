"""Gaussian outlier filter with kernel weighting for gaze data."""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class GaussianOutlierFilter:
    """
    Filters outliers using Gaussian statistics with kernel weighting.
    
    Maintains a sliding window of gaze points, computes statistics,
    rejects points outside 2σ, and applies Gaussian kernel weighting.
    """
    
    def __init__(
        self, 
        window_size: int = 10,
        std_threshold: float = 2.0,
        use_kernel_weighting: bool = True,
        kernel_sigma: float = 1.0
    ):
        """
        Initialize the Gaussian outlier filter.
        
        Args:
            window_size: Size of the sliding window for statistics (N gaze points)
            std_threshold: Number of standard deviations for outlier rejection (default: 2.0 for 2σ)
            use_kernel_weighting: Whether to apply Gaussian kernel weighting
            kernel_sigma: Standard deviation for Gaussian kernel weighting
        """
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.use_kernel_weighting = use_kernel_weighting
        self.kernel_sigma = kernel_sigma
        
        # Sliding window of last N gaze points
        self.gaze_history = deque(maxlen=window_size)
    
    def _compute_statistics(self) -> Tuple[float, float, float, float]:
        """
        Compute mean and standard deviation from the sliding window.
        
        Returns:
            Tuple of (x_mean, y_mean, x_std, y_std)
        """
        if len(self.gaze_history) == 0:
            return (0.0, 0.0, 1.0, 1.0)
        
        # Extract x and y coordinates
        x_coords = [point[0] for point in self.gaze_history]
        y_coords = [point[1] for point in self.gaze_history]
        
        # Compute mean and standard deviation
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # Avoid division by zero
        x_std = max(x_std, 1e-6)
        y_std = max(y_std, 1e-6)
        
        return (x_mean, y_mean, x_std, y_std)
    
    def _is_outlier(
        self,
        measurement: Tuple[float, float],
        x_mean: float,
        y_mean: float,
        x_std: float,
        y_std: float
    ) -> bool:
        """
        Check if measurement is outside 2σ threshold.
        
        Args:
            measurement: Tuple of (x, y) coordinates
            x_mean: Mean of x coordinates
            y_mean: Mean of y coordinates
            x_std: Standard deviation of x coordinates
            y_std: Standard deviation of y coordinates
            
        Returns:
            True if measurement is an outlier, False otherwise
        """
        x, y = measurement
        
        # Compute z-scores
        x_z_score = abs(x - x_mean) / x_std
        y_z_score = abs(y - y_mean) / y_std
        
        # Reject if outside 2σ (or configured threshold)
        return x_z_score > self.std_threshold or y_z_score > self.std_threshold
    
    def _apply_gaussian_kernel_weighting(
        self,
        measurement: Tuple[float, float],
        x_mean: float,
        y_mean: float,
        x_std: float,
        y_std: float
    ) -> Tuple[float, float]:
        """
        Apply Gaussian kernel weighting to the measurement.
        
        Points closer to the mean get higher weight, points further away get lower weight.
        
        Args:
            measurement: Tuple of (x, y) coordinates
            x_mean: Mean of x coordinates
            y_mean: Mean of y coordinates
            x_std: Standard deviation of x coordinates
            y_std: Standard deviation of y coordinates
            
        Returns:
            Weighted (x, y) coordinates
        """
        x, y = measurement
        
        # Compute normalized distances
        x_dist = (x - x_mean) / (x_std + 1e-6)
        y_dist = (y - y_mean) / (y_std + 1e-6)
        
        # Compute Gaussian kernel weights
        # Weight = exp(-0.5 * (distance / sigma)^2)
        x_weight = np.exp(-0.5 * (x_dist / self.kernel_sigma) ** 2)
        y_weight = np.exp(-0.5 * (y_dist / self.kernel_sigma) ** 2)
        
        # Weighted average: w * measurement + (1 - w) * mean
        weighted_x = x_weight * x + (1 - x_weight) * x_mean
        weighted_y = y_weight * y + (1 - y_weight) * y_mean
        
        return (weighted_x, weighted_y)
    
    def filter(self, measurement: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Filter a measurement using Gaussian statistics and kernel weighting.
        
        Pipeline:
        1. Maintain sliding window of last N gaze points
        2. Compute mean and standard deviation
        3. Reject points outside 2σ (or return None)
        4. Apply Gaussian kernel weighting
        5. Return filtered point
        
        Args:
            measurement: Tuple of (x, y) coordinates
            
        Returns:
            Filtered (x, y) coordinates, or None if measurement is rejected as outlier
        """
        x, y = measurement
        
        # Need at least 3 points to compute meaningful statistics
        if len(self.gaze_history) < 3:
            # Not enough data yet, accept all measurements
            self.gaze_history.append(measurement)
            return measurement
        
        # Compute mean and standard deviation from sliding window
        x_mean, y_mean, x_std, y_std = self._compute_statistics()
        
        # Apply Gaussian kernel weighting (or use measurement)
        # When point is outside 2σ, still return a smoothed value so cursor keeps moving;
        # returning None would freeze the cursor on valid gaze shifts.
        if self._is_outlier(measurement, x_mean, y_mean, x_std, y_std):
            # Outlier: blend toward mean so cursor moves smoothly instead of freezing
            if self.use_kernel_weighting:
                filtered_point = self._apply_gaussian_kernel_weighting(
                    measurement, x_mean, y_mean, x_std, y_std
                )
            else:
                # Pull toward mean to limit jump
                blend = 0.5
                filtered_point = (
                    blend * x_mean + (1 - blend) * x,
                    blend * y_mean + (1 - blend) * y,
                )
            self.gaze_history.append(measurement)
            return filtered_point

        if self.use_kernel_weighting:
            filtered_point = self._apply_gaussian_kernel_weighting(
                measurement, x_mean, y_mean, x_std, y_std
            )
        else:
            # No weighting, just return the measurement
            filtered_point = measurement
        
        # Add to sliding window history
        self.gaze_history.append(measurement)
        
        return filtered_point
    
    def get_statistics(self) -> Tuple[float, float, float, float]:
        """
        Get current statistics from the sliding window.
        
        Returns:
            Tuple of (x_mean, y_mean, x_std, y_std)
        """
        return self._compute_statistics()
    
    def get_window_size(self) -> int:
        """Get current window size."""
        return len(self.gaze_history)
    
    def set_std_threshold(self, threshold: float):
        """
        Update the standard deviation threshold for outlier rejection.
        
        Args:
            threshold: New threshold value (e.g., 2.0 for 2σ)
        """
        self.std_threshold = threshold
    
    def set_kernel_sigma(self, sigma: float):
        """
        Update the Gaussian kernel sigma parameter.
        
        Args:
            sigma: New sigma value for kernel weighting
        """
        self.kernel_sigma = sigma
    
    def reset(self):
        """Reset the filter state (clear sliding window)."""
        self.gaze_history.clear()
