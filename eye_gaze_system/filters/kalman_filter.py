"""2D Constant Velocity Kalman Filter for smoothing gaze coordinates."""

import numpy as np
from typing import Tuple, Optional


class KalmanFilter:
    """
    2D Constant Velocity Kalman Filter for smoothing gaze coordinates.
    
    State vector: [x, y, dx, dy]
    Measurement: [x, y]
    """
    
    def __init__(
        self, 
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_covariance: float = 1.0,
        dt: float = 1.0
    ):
        """
        Initialize the 2D constant velocity Kalman filter.
        
        Args:
            process_noise: Process noise covariance (Q) - controls how much the model
                          trusts the motion model vs measurements. Lower = more trust in model.
            measurement_noise: Measurement noise covariance (R) - controls how much the model
                              trusts measurements. Lower = more trust in measurements.
            initial_covariance: Initial state covariance (P0) - uncertainty in initial state
            dt: Time step (delta time) - typically 1.0 for normalized time or actual frame time
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance
        self.dt = dt
        
        # State vector: [x, y, dx, dy]
        # x, y: position coordinates
        # dx, dy: velocity components
        self.state = np.zeros(4, dtype=np.float64)
        
        # State covariance matrix (4x4)
        self.covariance = np.eye(4, dtype=np.float64) * initial_covariance
        
        # State transition matrix F (4x4) - constant velocity model
        # x' = x + dx*dt
        # y' = y + dy*dt
        # dx' = dx (constant velocity)
        # dy' = dy (constant velocity)
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Measurement matrix H (2x4) - we only observe position [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        
        # Process noise covariance matrix Q (4x4)
        # For constant velocity model, process noise affects both position and velocity
        # Using a standard model where noise is proportional to dt^2 for position
        # and dt for velocity
        q_pos = process_noise * dt * dt
        q_vel = process_noise * dt
        self.Q = np.array([
            [q_pos, 0.0,   0.0,   0.0  ],
            [0.0,   q_pos, 0.0,   0.0  ],
            [0.0,   0.0,   q_vel, 0.0  ],
            [0.0,   0.0,   0.0,   q_vel]
        ], dtype=np.float64)
        
        # Measurement noise covariance matrix R (2x2)
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        self.initialized = False
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict step: propagate state and covariance forward in time.
        
        Returns:
            Tuple of predicted (x, y) coordinates
        """
        if not self.initialized:
            return (0.0, 0.0)
        
        # Predict state: x_k|k-1 = F * x_k-1|k-1
        self.state = self.F @ self.state
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # Return predicted position
        return (float(self.state[0]), float(self.state[1]))
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update step: incorporate new measurement to refine state estimate.
        
        Args:
            measurement: Tuple of (x, y) measured coordinates
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        measurement_array = np.array([measurement[0], measurement[1]], dtype=np.float64)
        
        if not self.initialized:
            # Initialize state with first measurement
            self.state[0] = measurement[0]
            self.state[1] = measurement[1]
            self.state[2] = 0.0  # dx = 0
            self.state[3] = 0.0  # dy = 0
            self.initialized = True
            return measurement
        
        # Innovation (residual): y = z - H * x_k|k-1
        innovation = measurement_array - self.H @ self.state
        
        # Innovation covariance: S = H * P_k|k-1 * H^T + R
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Kalman gain: K = P_k|k-1 * H^T * S^-1
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K * y
        self.state = self.state + K @ innovation
        
        # Update covariance: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(4, dtype=np.float64)
        self.covariance = (I - K @ self.H) @ self.covariance
        
        # Return smoothed position
        return (float(self.state[0]), float(self.state[1]))
    
    def filter(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Complete filter step: predict then update.
        
        This is a convenience method that combines predict and update steps.
        
        Args:
            measurement: Tuple of (x, y) measured coordinates
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        # Predict
        self.predict()
        
        # Update with measurement
        return self.update(measurement)
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Get current state vector.
        
        Returns:
            Tuple of (x, y, dx, dy)
        """
        return (
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(self.state[3])
        )
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            Tuple of (dx, dy)
        """
        return (float(self.state[2]), float(self.state[3]))
    
    def set_process_noise(self, process_noise: float):
        """
        Update process noise parameter and recompute Q matrix.
        
        Args:
            process_noise: New process noise covariance value
        """
        self.process_noise = process_noise
        q_pos = process_noise * self.dt * self.dt
        q_vel = process_noise * self.dt
        self.Q = np.array([
            [q_pos, 0.0,   0.0,   0.0  ],
            [0.0,   q_pos, 0.0,   0.0  ],
            [0.0,   0.0,   q_vel, 0.0  ],
            [0.0,   0.0,   0.0,   q_vel]
        ], dtype=np.float64)
    
    def set_measurement_noise(self, measurement_noise: float):
        """
        Update measurement noise parameter and recompute R matrix.
        
        Args:
            measurement_noise: New measurement noise covariance value
        """
        self.measurement_noise = measurement_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
    
    def set_dt(self, dt: float):
        """
        Update time step and recompute F and Q matrices.
        
        Args:
            dt: New time step value
        """
        self.dt = dt
        
        # Update state transition matrix
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Update process noise matrix
        q_pos = self.process_noise * dt * dt
        q_vel = self.process_noise * dt
        self.Q = np.array([
            [q_pos, 0.0,   0.0,   0.0  ],
            [0.0,   q_pos, 0.0,   0.0  ],
            [0.0,   0.0,   q_vel, 0.0  ],
            [0.0,   0.0,   0.0,   q_vel]
        ], dtype=np.float64)
    
    def reset(self):
        """Reset the filter state to initial conditions."""
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * self.initial_covariance
        self.initialized = False
    
    def get_covariance(self) -> np.ndarray:
        """
        Get current state covariance matrix.
        
        Returns:
            4x4 covariance matrix
        """
        return self.covariance.copy()
