"""
Vision Bot Utilities

Contains coordinate validation, clamping functions, and other utilities
for the simplified vision bot.
"""
import math
from typing import List, Tuple


def clamp_coordinate(value: int, min_val: int, max_val: int) -> int:
    """
    Clamp a coordinate value to be within valid bounds.
    
    Args:
        value: The coordinate value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped coordinate value
    """
    return max(min_val, min(max_val, value))


def validate_and_clamp_coordinates(x: int, y: int, page_width: int, page_height: int) -> Tuple[int, int]:
    """
    Validate and clamp coordinates to ensure they're within the viewport.
    
    Args:
        x: X coordinate
        y: Y coordinate  
        page_width: Width of the viewport
        page_height: Height of the viewport
        
    Returns:
        Tuple of (clamped_x, clamped_y)
    """
    # Handle invalid/infinite values with safe fallbacks
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        x = 0
    if not isinstance(y, (int, float)) or not math.isfinite(y):
        y = 0
    
    # Clamp to viewport bounds (leave 1px margin from edges)
    x = clamp_coordinate(int(x), 0, page_width - 1)
    y = clamp_coordinate(int(y), 0, page_height - 1)
    
    return x, y

def get_gemini_box_2d_center_pixels(box_2d: List[int], page_width: int, page_height: int) -> Tuple[int, int]:
    """
    Get the center point of a Gemini box_2d in pixel coordinates.

    Args:
        box_2d: Gemini box_2d as [y_min, x_min, y_max, x_max] - could be normalized (0-1000) or pixel coordinates
        page_width: Page width in pixels
        page_height: Page height in pixels

    Returns:
        Tuple of (center_x, center_y) in pixel coordinates
    """
    if len(box_2d) != 4:
        return 0, 0

    y_min, x_min, y_max, x_max = box_2d

    # Ensure all coordinates are valid numbers
    if not all(isinstance(coord, (int, float)) and math.isfinite(coord)
               for coord in [y_min, x_min, y_max, x_max]):
        return 0, 0

    # Check if coordinates are already pixel coordinates (unlikely to be > 1000 for normalized)
    max_coord = max(abs(y_min), abs(x_min), abs(y_max), abs(x_max))
    if max_coord > 1000:
        # These are already pixel coordinates
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        return center_x, center_y

    # Convert normalized coordinates (0-1000) to pixels
    center_x_norm = (x_min + x_max) / 2
    center_y_norm = (y_min + y_max) / 2

    # Scale to actual pixel coordinates
    center_x = int(center_x_norm / 1000.0 * page_width)
    center_y = int(center_y_norm / 1000.0 * page_height)

    # Clamp to page bounds
    # center_x = clamp_coordinate(center_x, 0, page_width - 1)
    # center_y = clamp_coordinate(center_y, 0, page_height - 1)

    return center_x, center_y


