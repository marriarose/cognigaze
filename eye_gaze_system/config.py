"""Configuration module for eye gaze system. All hyperparameters are adjustable via config dictionary."""

from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration dictionary.
    All hyperparameters can be overridden by passing a modified config.
    """
    return {
        # System
        "use_filter": True,
        "use_calibration": True,
        "filter_type": "kalman",
        "min_confidence": 0.2,
        "target_fps": 30.0,
        "camera_index": 0,
        "disable_failsafe": True,

        # Face detection
        "refine_landmarks": True,

        # Iris tracker
        "use_both_eyes": False,
        "enable_fallback": True,

        # Gaze estimator
        "screen_distance": 1.0,
        "screen_width": 1.0,
        "screen_height": 1.0,

        # Outlier filter
        "outlier_window_size": 10,
        "outlier_std_threshold": 2.0,
        "outlier_use_kernel_weighting": True,
        "outlier_kernel_sigma": 1.0,

        # Kalman filter — tuned for raw iris coords (range ~0.015, not 0-1).
        # process_noise: how much the iris can accelerate between frames
        # measurement_noise: how noisy the MediaPipe iris reading is
        "kalman_process_noise": 1e-4,
        "kalman_measurement_noise": 1e-3,
        "kalman_dt": 1.0,

        # One Euro filter
        "one_euro_min_cutoff": 1.0,
        "one_euro_beta": 0.0,
        "one_euro_d_cutoff": 1.0,

        # Weighted average filter
        "weighted_alpha": 0.7,
        "weighted_window_size": 10,

        # Blink detector
        "ear_threshold": 0.25,
        "consecutive_frames": 3,
        "long_blink_frames": 10,
        "debounce_time": 0.3,

        # Tracking state
        "confidence_threshold": 0.5,
        "freeze_threshold": 0.3,
        "max_freeze_frames": 10,
        "stability_window": 5,

        # Cursor control
        "enable_interpolation": False,   # interpolation adds blocking sleep() calls per frame
        "interpolation_steps": 1,
        "cursor_min_move_px": 3,         # px threshold; keeps tremor from wiggling cursor

        # Camera
        "camera_flip_horizontal": True,
        "camera_use_threading": True,
        "camera_buffer_size": 2,

        # Calibration
        "calibration_file": "calibration_data.json",
        "max_frame_drop": 30,

        # Iris-to-screen mapping (raw iris coords, landmark 473).
        # Gains measured by diagnose_gaze.py: gain = 0.85 / travel
        #   X travel = 0.0215  →  gain_x = 39.5
        #   Y travel = 0.0117  →  gain_y = 72.8
        # X axis is inverted (camera flip): handled in main.py with -(iris_x - cx)
        # Run diagnose_gaze.py to remeasure if gains feel wrong.
        "iris_invert_x": True,           # looking right → iris_x DECREASES on flipped camera
        "relative_iris_gain": 39.5,
        "relative_iris_gain_x": 39.5,    # raise if can't reach left/right edges
        "relative_iris_gain_y": 72.8,    # raise if can't reach top/bottom edges
        "gaze_smoothing_alpha": 0.6,
        "smooth_fast_thresh_px": 80,     # px distance above which move is intentional (no smoothing)
        "smooth_slow_alpha": 0.35,       # smoothing weight when holding still (tremor damping)
        "gaze_dead_zone": 0.001,         # iris noise floor; lower = more sensitive

        # Debug: bypass filters/calibration and map iris pixel -> screen directly (with prints)
        "debug_direct_cursor": False,
    }


def merge_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user config with defaults. User values override defaults.
    """
    config = get_default_config()
    if user_config:
        for key, value in user_config.items():
            if key in config:
                config[key] = value
    return config


class Config:
    """
    Configuration container. Wraps config dict for type-safe access.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self._config = merge_config(config or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config