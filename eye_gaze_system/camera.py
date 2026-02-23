"""Camera handling module with multi-threaded video capture."""

import cv2
from typing import Optional, Tuple
import numpy as np
import threading
import time
from collections import deque


class Camera:
    """Handles camera initialization and frame capture with multi-threading support."""
    
    def __init__(self, camera_index: int = 0, flip_horizontal: bool = True, use_threading: bool = True, buffer_size: int = 2):
        """
        Initialize the camera.
        
        Args:
            camera_index: Index of the camera device (default: 0)
            flip_horizontal: Whether to flip frames horizontally (default: True)
            use_threading: Whether to use multi-threaded capture (default: True)
            buffer_size: Size of frame buffer for threaded capture (default: 2)
        """
        self.camera_index = camera_index
        self.flip_horizontal = flip_horizontal
        self.use_threading = use_threading
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Threading support
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_available = threading.Event()
    
    def _capture_loop(self):
        """Internal capture loop for threaded operation — always keeps the LATEST frame."""
        while self.running:
            if self.cap is None:
                time.sleep(0.005)
                continue
            success, frame = self.cap.read()
            if success:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                # Store directly; reader will .copy() under lock
                with self.lock:
                    self.latest_frame = frame
                    self.frame_available.set()
            # No sleep on failure — spin at camera rate to minimise lag
    
    def initialize(self) -> bool:
        """
        Initialize the camera capture.
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        
        # Minimize buffer so we always read the LATEST frame (no stale frame lag)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Request 30 fps from the driver
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # MJPEG codec: much faster USB bandwidth than raw YUV on most webcams
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if self.use_threading:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            # Wait for first frame
            self.frame_available.wait(timeout=2.0)
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self.cap is None:
            return False, None
        
        if self.use_threading:
            # Get latest frame from thread
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    return True, frame
                return False, None
        else:
            # Synchronous capture
            success, frame = self.cap.read()
            
            if not success:
                return False, None
            
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            return True, frame
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.cap is None:
            return 0, 0
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def release(self):
        """Release the camera resource."""
        if self.use_threading:
            self.running = False
            if self.capture_thread is not None:
                self.capture_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.latest_frame = None
            self.frame_buffer.clear()
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()