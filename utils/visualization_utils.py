"""
Visualization Utilities

Contains functions for drawing bounding boxes and other visual debugging utilities.
"""
import io
from typing import List

from PIL import Image, ImageDraw


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
