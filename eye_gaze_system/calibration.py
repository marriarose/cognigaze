"""Calibration module for gaze-to-screen mapping with 5-point calibration grid."""

from typing import Optional, Tuple, List
import numpy as np
import cv2
import json
import os
from collections import deque

try:
    import pyautogui
except ImportError:
    pyautogui = None


class Calibration:
    """
    Handles calibration for mapping gaze to screen coordinates.
    
    Uses 5-point calibration grid and affine transform.
    """
    
    # Default calibration file path
    DEFAULT_CALIBRATION_FILE = "calibration_data.json"
    
    # Number of calibration points
    NUM_CALIBRATION_POINTS = 5
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize the calibration.
        
        Args:
            calibration_file: Path to calibration file (default: calibration_data.json)
        """
        self.calibration_file = calibration_file or self.DEFAULT_CALIBRATION_FILE
        
        # Calibration data
        self.gaze_points: List[Tuple[float, float]] = []  # Normalized gaze coordinates
        self.screen_points: List[Tuple[int, int]] = []  # Screen coordinates
        self.calibrated = False
        
        # Affine transform matrix (2x3)
        self.transform_matrix: Optional[np.ndarray] = None
        
        # Calibration grid points (5 points: corners + center)
        self.grid_points = self._generate_calibration_grid()
        
        # Current calibration state
        self.current_point_index = 0
        self.collecting_samples = False
        self.sample_buffer: deque = deque(maxlen=30)  # Collect 30 samples per point
        self.samples_collected = 0
        
        # Load existing calibration if available
        self.load()
    
    def _generate_calibration_grid(self) -> List[Tuple[int, int]]:
        """
        Generate 5-point calibration grid (corners + center).
        
        Returns:
            List of (x, y) screen coordinates for calibration points
        """
        screen_w, screen_h = self._get_screen_size()
        
        # 5-point grid: corners + center
        margin = 0.1  # 10% margin from edges
        points = [
            (int(screen_w * margin), int(screen_h * margin)),  # Top-left
            (int(screen_w * (1 - margin)), int(screen_h * margin)),  # Top-right
            (int(screen_w * 0.5), int(screen_h * 0.5)),  # Center
            (int(screen_w * margin), int(screen_h * (1 - margin))),  # Bottom-left
            (int(screen_w * (1 - margin)), int(screen_h * (1 - margin)))  # Bottom-right
        ]
        
        return points
    
    def _get_screen_size(self) -> Tuple[int, int]:
        """Get screen size."""
        if pyautogui is not None:
            try:
                return pyautogui.size()
            except:
                pass
        return (1920, 1080)  # Default fallback
    
    def start_calibration(self):
        """Start calibration process."""
        self.gaze_points.clear()
        self.screen_points.clear()
        self.current_point_index = 0
        self.collecting_samples = False
        self.sample_buffer.clear()
        self.samples_collected = 0
        self.calibrated = False
        self.transform_matrix = None
    
    def get_current_calibration_point(self) -> Optional[Tuple[int, int]]:
        """
        Get the current calibration point to display.
        
        Returns:
            (x, y) screen coordinates of current calibration point, or None if done
        """
        if self.current_point_index >= len(self.grid_points):
            return None
        return self.grid_points[self.current_point_index]
    
    def collect_gaze_sample(self, gaze_point: Tuple[float, float]):
        """
        Collect a gaze sample for the current calibration point.
        
        Args:
            gaze_point: Normalized gaze coordinates [0, 1]
        """
        if self.current_point_index >= len(self.grid_points):
            return
        
        if not self.collecting_samples:
            self.collecting_samples = True
            self.sample_buffer.clear()
            self.samples_collected = 0
        
        # Add sample to buffer
        self.sample_buffer.append(gaze_point)
        self.samples_collected += 1
    
    def finish_current_point(self) -> bool:
        """
        Finish collecting samples for current point and move to next.
        
        Returns:
            True if more points remain, False if calibration complete
        """
        if self.current_point_index >= len(self.grid_points):
            return False
        
        if len(self.sample_buffer) == 0:
            return True
        
        # Compute average gaze point from samples
        gaze_array = np.array(list(self.sample_buffer))
        avg_gaze = np.mean(gaze_array, axis=0)
        
        # Get corresponding screen point
        screen_point = self.grid_points[self.current_point_index]
        
        # Store calibration pair
        self.gaze_points.append((float(avg_gaze[0]), float(avg_gaze[1])))
        self.screen_points.append(screen_point)
        
        # Move to next point
        self.current_point_index += 1
        self.collecting_samples = False
        self.sample_buffer.clear()
        self.samples_collected = 0
        
        # Check if calibration complete
        if self.current_point_index >= len(self.grid_points):
            # Compute affine transform
            return self.calibrate()
        
        return True
    
    def calibrate(self) -> bool:
        """
        Compute affine transform from collected calibration points.
        
        Returns:
            True if calibration successful, False otherwise
        """
        if len(self.gaze_points) < 3:
            return False
        
        # Convert to numpy arrays
        gaze_array = np.array(self.gaze_points, dtype=np.float32)
        screen_array = np.array(self.screen_points, dtype=np.float32)
        
        # Compute affine transform
        # Affine transform: [x', y'] = [x, y, 1] * M
        # where M is a 3x2 matrix (we use 2x3 for cv2 compatibility)
        
        # Use cv2.getAffineTransform for 3 points, or estimateAffine2D for more
        if len(self.gaze_points) == 3:
            # Exactly 3 points - use getAffineTransform
            self.transform_matrix = cv2.getAffineTransform(
                gaze_array[:3],
                screen_array[:3]
            )
        else:
            # 4+ points - use estimateAffine2D (more robust)
            transform_result = cv2.estimateAffine2D(
                gaze_array,
                screen_array,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            if transform_result is not None:
                self.transform_matrix = transform_result[0]
            else:
                return False
        
        self.calibrated = True
        return True
    
    def map_to_screen(
        self, 
        gaze_point: Tuple[float, float],
        screen_width: int,
        screen_height: int
    ) -> Tuple[int, int]:
        """
        Map a gaze point to screen coordinates using affine transform.
        
        Args:
            gaze_point: Normalized gaze coordinates [0, 1]
            screen_width: Screen width (for bounds checking)
            screen_height: Screen height (for bounds checking)
            
        Returns:
            Screen coordinates (x, y)
        """
        if not self.calibrated or self.transform_matrix is None:
            # Fallback to direct mapping
            return int(gaze_point[0] * screen_width), int(gaze_point[1] * screen_height)
        
        # Apply affine transform
        gaze_array = np.array([[gaze_point[0], gaze_point[1]]], dtype=np.float32)
        screen_point = cv2.transform(gaze_array, self.transform_matrix)[0][0]
        
        screen_x = int(screen_point[0])
        screen_y = int(screen_point[1])
        
        # Clamp to screen bounds (ensure cursor stays within screen)
        screen_x = max(0, min(screen_x, screen_width - 1))
        screen_y = max(0, min(screen_y, screen_height - 1))
        
        return screen_x, screen_y
    
    def draw_calibration_grid(self, frame: np.ndarray, frame_width: int, frame_height: int):
        """
        Draw calibration grid overlay on frame.
        
        Args:
            frame: Frame to draw on
            frame_width: Frame width
            frame_height: Frame height
        """
        if self.current_point_index >= len(self.grid_points):
            return
        
        # Get current calibration point
        screen_point = self.get_current_calibration_point()
        if screen_point is None:
            return
        
        # Draw all grid points
        for i, point in enumerate(self.grid_points):
            # Map screen coordinates to frame coordinates for visualization
            # This is approximate - we show where user should look
            x = int(point[0] * frame_width / self._get_screen_size()[0])
            y = int(point[1] * frame_height / self._get_screen_size()[1])
            
            if i < self.current_point_index:
                # Completed points - green
                cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
            elif i == self.current_point_index:
                # Current point - red, pulsing
                radius = 20 + int(5 * np.sin(cv2.getTickCount() / 1000.0))
                cv2.circle(frame, (x, y), radius, (0, 0, 255), 3)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            else:
                # Upcoming points - gray
                cv2.circle(frame, (x, y), 10, (128, 128, 128), 1)
        
        # Draw progress text
        progress_text = f"Calibration: {self.current_point_index + 1}/{len(self.grid_points)}"
        if self.collecting_samples:
            progress_text += f" (Samples: {self.samples_collected})"
        cv2.putText(
            frame,
            progress_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    def save(self, file_path: Optional[str] = None) -> bool:
        """
        Save calibration data to file.
        
        Args:
            file_path: Path to save file (uses default if None)
            
        Returns:
            True if save successful, False otherwise
        """
        if not self.calibrated or self.transform_matrix is None:
            return False
        
        file_path = file_path or self.calibration_file
        
        try:
            calibration_data = {
                'gaze_points': self.gaze_points,
                'screen_points': self.screen_points,
                'transform_matrix': self.transform_matrix.tolist(),
                'calibrated': self.calibrated
            }
            
            with open(file_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load(self, file_path: Optional[str] = None) -> bool:
        """
        Load calibration data from file.
        
        Args:
            file_path: Path to load file (uses default if None)
            
        Returns:
            True if load successful, False otherwise
        """
        file_path = file_path or self.calibration_file
        
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'r') as f:
                calibration_data = json.load(f)
            
            self.gaze_points = [tuple(p) for p in calibration_data['gaze_points']]
            self.screen_points = [tuple(p) for p in calibration_data['screen_points']]
            self.transform_matrix = np.array(calibration_data['transform_matrix'], dtype=np.float32)
            self.calibrated = calibration_data.get('calibrated', False)
            
            return self.calibrated
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def reset(self):
        """Reset calibration data."""
        self.gaze_points.clear()
        self.screen_points.clear()
        self.calibrated = False
        self.transform_matrix = None
        self.current_point_index = 0
        self.collecting_samples = False
        self.sample_buffer.clear()
        self.samples_collected = 0
        
        # Delete calibration file if it exists
        if os.path.exists(self.calibration_file):
            try:
                os.remove(self.calibration_file)
            except:
                pass
    
    def get_samples_needed(self) -> int:
        """
        Get number of samples needed for current point.
        
        Returns:
            Number of samples needed
        """
        return 30 - self.samples_collected
    
    def is_collecting(self) -> bool:
        """Check if currently collecting samples."""
        return self.collecting_samples
    
    def is_calibration_complete(self) -> bool:
        """Check if calibration is complete."""
        return self.calibrated