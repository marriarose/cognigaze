"""Utility functions for the eye gaze system."""

import numpy as np
from typing import Tuple, Optional


def normalize_coordinates(
    x: float, 
    y: float, 
    frame_width: int, 
    frame_height: int
) -> Tuple[float, float]:
    """
    Normalize coordinates from frame space to [0, 1] range.
    
    Args:
        x: X coordinate in frame space
        y: Y coordinate in frame space
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Tuple of normalized (x, y) coordinates
    """
    return x / frame_width, y / frame_height


def denormalize_coordinates(
    x: float, 
    y: float, 
    frame_width: int, 
    frame_height: int
) -> Tuple[int, int]:
    """
    Convert normalized coordinates [0, 1] to frame space.
    
    Args:
        x: Normalized x coordinate [0, 1]
        y: Normalized y coordinate [0, 1]
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        Tuple of (x, y) coordinates in frame space
    """
    return int(x * frame_width), int(y * frame_height)


def map_to_screen(
    x: float, 
    y: float, 
    screen_width: int, 
    screen_height: int
) -> Tuple[int, int]:
    """
    Map normalized coordinates [0, 1] to screen coordinates with boundary clamping.
    
    Args:
        x: Normalized x coordinate [0, 1]
        y: Normalized y coordinate [0, 1]
        screen_width: Width of the screen
        screen_height: Height of the screen
        
    Returns:
        Tuple of (x, y) screen coordinates clamped to screen bounds
    """
    # Clamp normalized coordinates to [0, 1]
    x = clamp(x, 0.0, 1.0)
    y = clamp(y, 0.0, 1.0)
    
    # Map to screen coordinates
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    
    # Clamp to screen boundaries
    screen_x = max(0, min(screen_x, screen_width - 1))
    screen_y = max(0, min(screen_y, screen_height - 1))
    
    return screen_x, screen_y


def clamp_screen_coordinates(
    x: int,
    y: int,
    screen_width: int,
    screen_height: int
) -> Tuple[int, int]:
    """
    Clamp screen coordinates to screen boundaries.
    
    Args:
        x: X coordinate
        y: Y coordinate
        screen_width: Width of the screen
        screen_height: Height of the screen
        
    Returns:
        Tuple of clamped (x, y) screen coordinates
    """
    x = max(0, min(x, screen_width - 1))
    y = max(0, min(y, screen_height - 1))
    return x, y


def calculate_distance(
    point1: Tuple[float, float], 
    point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))
