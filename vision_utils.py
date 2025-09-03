"""
Vision Bot Utilities

Contains coordinate validation, clamping functions, and other utilities
for the simplified vision bot.
"""
import io
import math
from typing import List, Tuple

from PIL import Image, ImageDraw


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
        box_2d: Gemini box_2d as [y_min, x_min, y_max, x_max] normalized 0-1000
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
    
    # Convert normalized coordinates (0-1000) to pixels
    center_x_norm = (x_min + x_max) / 2
    center_y_norm = (y_min + y_max) / 2
    
    # Scale to actual pixel coordinates
    center_x = int(center_x_norm / 1000.0 * page_width)
    center_y = int(center_y_norm / 1000.0 * page_height)
    
    # Clamp to page bounds
    center_x = clamp_coordinate(center_x, 0, page_width - 1)
    center_y = clamp_coordinate(center_y, 0, page_height - 1)
    
    return center_x, center_y


def draw_bounding_boxes(screenshot: bytes, elements: List, save_path: str = None) -> bytes:
    """
    Draw bounding boxes on screenshot for debugging.
    
    Args:
        screenshot: Screenshot image as bytes
        elements: List of DetectedElement objects with box_2d attribute
        save_path: Optional path to save the image
        
    Returns:
        Modified image as bytes
    """
    try:
        image = Image.open(io.BytesIO(screenshot))
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        for i, element in enumerate(elements):
            if not hasattr(element, 'box_2d') or not element.box_2d:
                continue
                
            # Convert Gemini box_2d [y_min, x_min, y_max, x_max] (0-1000) to pixels
            y_min, x_min, y_max, x_max = element.box_2d
            
            # Scale from 0-1000 to actual image dimensions
            pixel_x_min = int(x_min / 1000.0 * img_width)
            pixel_y_min = int(y_min / 1000.0 * img_height)
            pixel_x_max = int(x_max / 1000.0 * img_width)
            pixel_y_max = int(y_max / 1000.0 * img_height)
            
            # Choose color based on element properties
            if hasattr(element, 'is_clickable') and element.is_clickable:
                color = "red"  # Clickable elements in red
            else:
                color = "blue"  # Non-clickable in blue
            
            # Draw bounding box
            draw.rectangle([pixel_x_min, pixel_y_min, pixel_x_max, pixel_y_max], outline=color, width=2)
            
            # Draw element index
            draw.text((pixel_x_min + 2, pixel_y_min + 2), str(i), fill=color)
            
            # Draw confidence if available
            if hasattr(element, 'confidence'):
                conf_text = f"{element.confidence:.2f}"
                draw.text((pixel_x_min + 2, pixel_y_max - 15), conf_text, fill=color)
        
        # Save to BytesIO
        output = io.BytesIO()
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
        
        # Optionally save to file
        if save_path:
            with open(save_path, "wb") as f:
                f.write(image_bytes)
        
        return image_bytes
        
    except Exception as e:
        print(f"⚠️ Error drawing bounding boxes: {e}")
        return screenshot  # Return original if drawing fails
