"""Cursor control layer for smooth cursor movement."""

import pyautogui
from typing import Optional, Tuple
import time
import math


class CursorController:
    """
    Handles cursor movement with smooth interpolation.
    
    This is the ONLY place where pyautogui.moveTo() is called.
    """
    
    def __init__(
        self,
        enable_interpolation: bool = True,
        interpolation_steps: int = 5,
        disable_failsafe: bool = True,
        min_move_px: int = 8,
    ):
        """
        Initialize cursor controller.
        
        Args:
            enable_interpolation: Enable smooth interpolation (default: True)
            interpolation_steps: Number of interpolation steps (default: 5)
            disable_failsafe: Disable PyAutoGUI failsafe (default: True)
            min_move_px: Don't move if distance to last position is below this (reduces jitter)
        """
        if disable_failsafe:
            pyautogui.FAILSAFE = False
        # pyautogui.PAUSE defaults to 0.1s — adds 100ms to EVERY moveTo call.
        # This is the primary source of cursor lag in gaze control.
        pyautogui.PAUSE = 0.0
        
        self.enable_interpolation = enable_interpolation
        self.interpolation_steps = interpolation_steps
        self.min_move_px = max(0, min_move_px)
        self.last_position: Optional[Tuple[int, int]] = None
        self._last_sent_position: Optional[Tuple[int, int]] = None
        self.last_move_time = time.time()
    
    def move_to(
        self,
        screen_x: int,
        screen_y: int,
        duration: float = 0.0
    ):
        """
        Move cursor to screen coordinates.

        Interpolation is intentionally disabled for gaze control: each gaze
        frame already carries the correct target position, and interpolation
        adds multiple blocking pyautogui.moveTo() calls + sleep() per frame,
        which directly causes the lag the user experiences.
        """
        # Skip sub-threshold moves to dampen tremor without adding lag
        if self.min_move_px > 0 and self._last_sent_position is not None:
            dx = screen_x - self._last_sent_position[0]
            dy = screen_y - self._last_sent_position[1]
            if math.sqrt(dx * dx + dy * dy) < self.min_move_px:
                self.last_position = self._last_sent_position
                return

        # Single direct move — no interpolation loop, no sleep
        pyautogui.moveTo(screen_x, screen_y)

        self.last_position = (screen_x, screen_y)
        self._last_sent_position = (screen_x, screen_y)
        self.last_move_time = time.time()
    
    def _interpolated_move_from(
        self, target_x: int, target_y: int, start_x: int, start_y: int
    ):
        """
        Move cursor with smooth interpolation from (start_x, start_y) to target.
        """
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 5:
            pyautogui.moveTo(target_x, target_y)
            return
        steps = self.interpolation_steps
        for i in range(1, steps + 1):
            t = i / steps
            eased_t = self._ease_in_out_quad(t)
            current_x = int(start_x + dx * eased_t)
            current_y = int(start_y + dy * eased_t)
            pyautogui.moveTo(current_x, current_y)
            time.sleep(0.001)

    def _interpolated_move(self, target_x: int, target_y: int):
        """
        Move cursor with smooth interpolation between last sent position and target.
        """
        start = self._last_sent_position if self._last_sent_position is not None else self.last_position
        if start is None:
            pyautogui.moveTo(target_x, target_y)
            return
        self._interpolated_move_from(target_x, target_y, start[0], start[1])
    
    def _ease_in_out_quad(self, t: float) -> float:
        """
        Quadratic ease-in-out function for smooth interpolation.
        
        Args:
            t: Time parameter [0, 1]
            
        Returns:
            Eased time parameter
        """
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
    
    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """
        Get last cursor position.
        
        Returns:
            Last position (x, y) or None
        """
        return self.last_position
    
    def reset(self):
        """Reset cursor controller state."""
        self.last_position = None
        self.last_move_time = time.time()