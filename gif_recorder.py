"""
GIF Recording Module for Bot Interactions

This module provides functionality to record bot interactions as high-resolution GIFs
with visual highlights showing what elements are being interacted with.
"""

import os
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import io
from playwright.sync_api import Page


@dataclass
class InteractionFrame:
    """Represents a single frame in the GIF recording"""
    timestamp: float
    screenshot: bytes
    interaction_type: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    element_box: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    action_description: Optional[str] = None
    highlight_color: str = "#00ffae"
    duration_ms: int = 800  # How long this frame should be displayed


class GIFRecorder:
    """Records bot interactions as animated GIFs with visual highlights"""
    
    def __init__(self, page: Page, output_dir: str = "gif_recordings"):
        self.page = page
        self.output_dir = output_dir
        self.frames: List[InteractionFrame] = []
        self.recording = False
        self.session_id = str(uuid.uuid4())[:8]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Default GIF settings
        self.default_frame_duration = 800  # milliseconds
        self.highlight_duration = 300  # milliseconds for highlight frames
        self.final_pause_duration = 2000  # milliseconds for final frame
        
    def start_recording(self) -> None:
        """Start recording interactions"""
        self.recording = True
        self.frames.clear()
        print(f"🎬 Started GIF recording session: {self.session_id}")
        
        # Capture initial frame
        self._capture_frame("Initial state")
        
    def stop_recording(self) -> str:
        """Stop recording and generate GIF"""
        if not self.recording:
            return None
            
        self.recording = False
        
        # Capture final frame with longer duration
        final_frame = self._capture_frame("Final state")
        if final_frame:
            final_frame.duration_ms = self.final_pause_duration
            
        # Generate GIF
        gif_path = self._generate_gif()
        
        print(f"🎬 GIF recording completed: {gif_path}")
        return gif_path
        
    def record_interaction(
        self,
        interaction_type: str,
        coordinates: Optional[Tuple[int, int]] = None,
        element_box: Optional[Tuple[int, int, int, int]] = None,
        action_description: Optional[str] = None,
        text_input: Optional[str] = None,
        keys_pressed: Optional[str] = None
    ) -> None:
        """Record an interaction with visual highlights"""
        if not self.recording:
            return
            
        # Create description based on interaction type
        if not action_description:
            if interaction_type == "click":
                action_description = f"Click at ({coordinates[0]}, {coordinates[1]})" if coordinates else "Click"
            elif interaction_type == "type":
                action_description = f"Type: '{text_input}'" if text_input else "Type text"
            elif interaction_type == "press":
                action_description = f"Press: {keys_pressed}" if keys_pressed else "Press key"
            elif interaction_type == "scroll":
                action_description = "Scroll page"
            else:
                action_description = f"{interaction_type.title()} action"
        
        # Capture pre-interaction frame
        self._capture_frame(
            f"Before: {action_description}",
            interaction_type=interaction_type,
            coordinates=coordinates,
            element_box=element_box,
            highlight_color="#ff6b6b",  # Red for "before"
            duration_ms=self.highlight_duration
        )
        
        # Small delay to ensure interaction completes
        time.sleep(0.1)
        
        # Capture post-interaction frame
        self._capture_frame(
            f"After: {action_description}",
            interaction_type=interaction_type,
            coordinates=coordinates,
            element_box=element_box,
            highlight_color="#4ecdc4",  # Teal for "after"
            duration_ms=self.highlight_duration
        )
        
    def _capture_frame(
        self,
        description: str,
        interaction_type: Optional[str] = None,
        coordinates: Optional[Tuple[int, int]] = None,
        element_box: Optional[Tuple[int, int, int, int]] = None,
        highlight_color: str = "#00ffae",
        duration_ms: int = None
    ) -> Optional[InteractionFrame]:
        """Capture a screenshot with optional highlights"""
        try:
            # Take screenshot
            screenshot_bytes = self.page.screenshot(type="png", full_page=False)
            
            # Create frame
            frame = InteractionFrame(
                timestamp=time.time(),
                screenshot=screenshot_bytes,
                interaction_type=interaction_type,
                coordinates=coordinates,
                element_box=element_box,
                action_description=description,
                highlight_color=highlight_color,
                duration_ms=duration_ms or self.default_frame_duration
            )
            
            self.frames.append(frame)
            return frame
            
        except Exception as e:
            print(f"⚠️ Failed to capture frame: {e}")
            return None
            
    def _generate_gif(self) -> str:
        """Generate GIF from captured frames"""
        if not self.frames:
            print("⚠️ No frames to generate GIF from")
            return None
            
        try:
            # Load and process frames
            pil_frames = []
            durations = []
            
            for frame in self.frames:
                # Load screenshot
                screenshot_img = Image.open(io.BytesIO(frame.screenshot))
                
                # Add highlights if specified
                if frame.coordinates or frame.element_box:
                    highlighted_img = self._add_highlights(
                        screenshot_img,
                        frame.coordinates,
                        frame.element_box,
                        frame.highlight_color,
                        frame.action_description
                    )
                else:
                    highlighted_img = screenshot_img
                
                pil_frames.append(highlighted_img)
                durations.append(frame.duration_ms)
            
            # Generate GIF filename
            timestamp = int(time.time())
            gif_filename = f"bot_interaction_{self.session_id}_{timestamp}.gif"
            gif_path = os.path.join(self.output_dir, gif_filename)
            
            # Save as GIF with high quality
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,  # Infinite loop
                optimize=True,
                quality=95
            )
            
            return gif_path
            
        except Exception as e:
            print(f"⚠️ Failed to generate GIF: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _add_highlights(
        self,
        img: Image.Image,
        coordinates: Optional[Tuple[int, int]],
        element_box: Optional[Tuple[int, int, int, int]],
        color: str,
        description: str
    ) -> Image.Image:
        """Add visual highlights to an image"""
        try:
            # Create a copy to avoid modifying the original
            highlighted_img = img.copy()
            draw = ImageDraw.Draw(highlighted_img)
            
            # Add element box highlight if provided
            if element_box:
                x1, y1, x2, y2 = element_box
                # Draw rectangle outline
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Add semi-transparent overlay
                overlay = Image.new('RGBA', highlighted_img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x1, y1, x2, y2], fill=(*self._hex_to_rgb(color), 50))
                highlighted_img = Image.alpha_composite(highlighted_img.convert('RGBA'), overlay).convert('RGB')
            
            # Add coordinate highlight if provided
            if coordinates:
                x, y = coordinates
                radius = 15
                
                # Draw circle
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=color,
                    width=4,
                    fill=(*self._hex_to_rgb(color), 100)
                )
                
                # Draw crosshair
                crosshair_size = 20
                draw.line([x - crosshair_size, y, x + crosshair_size, y], fill=color, width=2)
                draw.line([x, y - crosshair_size, x, y + crosshair_size], fill=color, width=2)
            
            # Add description text if provided
            if description:
                self._add_description_text(highlighted_img, description, color)
            
            return highlighted_img
            
        except Exception as e:
            print(f"⚠️ Failed to add highlights: {e}")
            return img
            
    def _add_description_text(self, img: Image.Image, text: str, color: str) -> None:
        """Add description text to the image"""
        try:
            draw = ImageDraw.Draw(img)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
            
            # Position text at top of image
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Add background rectangle for text
            padding = 10
            bg_x1 = 10
            bg_y1 = 10
            bg_x2 = bg_x1 + text_width + padding * 2
            bg_y2 = bg_y1 + text_height + padding * 2
            
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 180))
            
            # Add text
            text_x = bg_x1 + padding
            text_y = bg_y1 + padding
            draw.text((text_x, text_y), text, fill=color, font=font)
            
        except Exception as e:
            print(f"⚠️ Failed to add description text: {e}")
            
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
    def get_frame_count(self) -> int:
        """Get the current number of captured frames"""
        return len(self.frames)
        
    def clear_frames(self) -> None:
        """Clear all captured frames"""
        self.frames.clear()
        
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording

