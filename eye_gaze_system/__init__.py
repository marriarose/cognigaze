"""Eye Gaze System package."""

from .main import EyeGazeSystem, main
from .config import Config, get_default_config, merge_config
from .camera import Camera
from .face_landmarks import FaceLandmarkDetector
from .blink_detector import BlinkDetector
from .calibration import Calibration          # original 5-point calibration (unused by default)
from .gaze_calibration import GazeCalibration, CalibrationSession  # new 9-point calibration
from .cursor_control import CursorController
from .tracking_state import TrackingState

__all__ = [
    'EyeGazeSystem', 'main',
    'Config', 'get_default_config', 'merge_config',
    'Camera',
    'FaceLandmarkDetector',
    'BlinkDetector',
    'Calibration',
    'GazeCalibration', 'CalibrationSession',
    'CursorController',
    'TrackingState',
]