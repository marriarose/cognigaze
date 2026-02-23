"""Geometric gaze estimation module using 3D ray-plane intersection."""

from typing import Optional, Tuple, List
import numpy as np
from .iris_tracker import IrisTracker
from .face_landmarks import FaceLandmarkDetector


class GazeEstimator:
    """Estimates gaze direction using geometric 3D ray-plane intersection."""
    
    def __init__(
        self, 
        iris_tracker: IrisTracker,
        face_detector: FaceLandmarkDetector,
        screen_distance: float = 1.0,
        screen_width: float = 1.0,
        screen_height: float = 1.0
    ):
        """
        Initialize the geometric gaze estimator.
        
        Args:
            iris_tracker: IrisTracker instance
            face_detector: FaceLandmarkDetector instance
            screen_distance: Distance to screen plane in camera coordinate space (default: 1.0)
            screen_width: Screen width in camera coordinate space (default: 1.0)
            screen_height: Screen height in camera coordinate space (default: 1.0)
        """
        self.iris_tracker = iris_tracker
        self.face_detector = face_detector
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize screen plane transform matrix (4x4 homogeneous)
        # Screen plane is at z = screen_distance (w=0 plane assumption means z=0 in screen space)
        # We'll define the screen plane in camera coordinates
        self._setup_screen_plane()
    
    def _setup_screen_plane(self):
        """
        Set up the screen plane in camera coordinate space.
        
        The screen plane is defined as z = screen_distance in camera coordinates.
        For w=0 plane assumption, we use a plane equation: n·P + d = 0
        where n is the normal vector and d is the offset.
        """
        # Screen plane normal (pointing towards camera, z-axis)
        # In camera coordinates, screen is at z = screen_distance
        self.screen_normal = np.array([0.0, 0.0, 1.0])  # Normal pointing along z-axis
        
        # Screen plane point (center of screen at z = screen_distance)
        self.screen_plane_point = np.array([0.0, 0.0, self.screen_distance])
        
        # Plane equation: n·(P - P0) = 0, or n·P - n·P0 = 0
        # So d = -n·P0
        self.plane_d = -np.dot(self.screen_normal, self.screen_plane_point)
    
    def _construct_gaze_vector(
        self,
        landmarks: List,
        eye: str = 'left'
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Construct 3D gaze vector from iris center and eye socket center.
        
        Args:
            landmarks: List of facial landmark objects
            eye: Which eye ('left' or 'right')
            
        Returns:
            Tuple of (origin, direction) where:
                - origin: 3D point (eye socket center) as numpy array
                - direction: 3D normalized direction vector as numpy array
            Or None if construction fails
        """
        # Get iris center (3D normalized coordinates)
        iris_center_3d = self.face_detector.get_iris_center_3d(landmarks, eye)
        if iris_center_3d is None:
            return None
        
        # Get eye socket center (centroid of eye contour)
        socket_center = self.face_detector.get_eye_socket_center(landmarks, eye)
        if socket_center is None:
            return None
        
        # Convert to numpy arrays
        iris_point = np.array([iris_center_3d[0], iris_center_3d[1], iris_center_3d[2]])
        socket_point = np.array([socket_center[0], socket_center[1], socket_center[2]])
        
        # Gaze vector direction: from socket center to iris center
        # This represents the direction the eye is looking
        direction = iris_point - socket_point
        
        # Normalize direction vector
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        
        direction = direction / norm
        
        # Origin is the eye socket center
        origin = socket_point
        
        return (origin, direction)
    
    def _ray_plane_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute ray-plane intersection.
        
        Ray equation: P = O + t*D
        Plane equation: n·P + d = 0
        
        Solving for t:
        n·(O + t*D) + d = 0
        n·O + t*(n·D) + d = 0
        t = -(n·O + d) / (n·D)
        
        Args:
            origin: Ray origin point (3D)
            direction: Ray direction vector (3D, normalized)
            
        Returns:
            3D intersection point as numpy array, or None if no intersection
        """
        # Compute dot products
        n_dot_d = np.dot(self.screen_normal, direction)
        
        # Check if ray is parallel to plane
        if abs(n_dot_d) < 1e-6:
            return None
        
        # Compute intersection parameter t
        n_dot_o = np.dot(self.screen_normal, origin)
        t = -(n_dot_o + self.plane_d) / n_dot_d
        
        # Check if intersection is in front of camera (positive t)
        if t < 0:
            return None
        
        # Compute intersection point
        intersection = origin + t * direction
        
        return intersection
    
    def _project_to_screen_coordinates(
        self,
        intersection_3d: np.ndarray
    ) -> Tuple[float, float]:
        """
        Project 3D intersection point to 2D screen coordinates.
        
        MediaPipe landmarks (and thus ray-plane intersection) use normalized
        image coordinates where x, y are in [0, 1]. We use them directly as
        normalized screen coordinates and clamp to [0, 1].
        
        Args:
            intersection_3d: 3D point (x, y, z) in MediaPipe normalized space
            
        Returns:
            Tuple of normalized (x, y) screen coordinates [0, 1]
        """
        # MediaPipe space: x, y in [0, 1]; use directly as normalized screen coords
        normalized_x = max(0.0, min(1.0, float(intersection_3d[0])))
        normalized_y = max(0.0, min(1.0, float(intersection_3d[1])))
        return (normalized_x, normalized_y)
    
    def estimate_gaze(
        self, 
        landmarks: Optional[List],
        frame: Optional[np.ndarray] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        eye: str = 'left'
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate gaze direction using geometric 3D ray-plane intersection.
        
        This method does NOT apply any filtering - returns raw projected coordinates.
        
        Args:
            landmarks: List of facial landmark objects (can be None)
            frame: Full frame (BGR format) - required for fallback
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            eye: Which eye ('left' or 'right')
            
        Returns:
            Tuple of normalized (x, y) gaze coordinates, or None
        """
        # If landmarks are not available, try fallback
        if landmarks is None:
            if frame is not None:
                # Use fallback tracking method
                iris_position = self.iris_tracker.track(
                    landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    frame=frame
                )
                return iris_position
            return None
        
        # Construct 3D gaze vector from iris center and eye socket center
        gaze_data = self._construct_gaze_vector(landmarks, eye)
        
        if gaze_data is None:
            # Fallback to simple iris tracking if geometric method fails
            if frame is not None:
                iris_position = self.iris_tracker.track(
                    landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    frame=frame
                )
                return iris_position
            return None
        
        origin, direction = gaze_data
        
        # Compute ray-plane intersection
        intersection_3d = self._ray_plane_intersection(origin, direction)
        
        if intersection_3d is None:
            # Fallback to simple iris tracking if intersection fails
            if frame is not None:
                iris_position = self.iris_tracker.track(
                    landmarks,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    frame=frame
                )
                return iris_position
            return None
        
        # Project to 2D screen coordinates
        screen_coords = self._project_to_screen_coordinates(intersection_3d)
        
        return screen_coords
    
    def set_screen_parameters(
        self,
        screen_distance: float,
        screen_width: float,
        screen_height: float
    ):
        """
        Update screen plane parameters.
        
        Args:
            screen_distance: Distance to screen plane in camera coordinate space
            screen_width: Screen width in camera coordinate space
            screen_height: Screen height in camera coordinate space
        """
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._setup_screen_plane()
