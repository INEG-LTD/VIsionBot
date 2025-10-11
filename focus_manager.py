"""
Focus Manager for UI element focusing and context management.

Architecture:
- Overlay manager labels all visible, interactable elements with ephemeral indices (0..N)
- AI receives: natural-language intent + screenshot + elements[] (hybrid vision-first with DOM assist)
- AI selects index(es) that best satisfy the intent
- Returns no-focus if confidence is low rather than guessing
- Maintains short-term memory to avoid re-selecting recent failures
- Ranks by: semantic match → role/affordance → proximity/association → visual cues
"""

from __future__ import annotations

import os
import traceback
import uuid
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from playwright.sync_api import Page

from element_detection.overlay_manager import OverlayManager
from utils import PageUtils
from ai_utils import generate_text
from utils.vision_resolver import (
    resolve_click_from_overlays,
    resolve_field_from_overlays,
    resolve_select_from_overlays,
    resolve_datetime_from_overlays,
    resolve_upload_from_overlays,
)
from utils.intent_parsers import parse_action_intent
from interaction_deduper import InteractionDeduper


@dataclass
class FocusedElement:
    """Represents a focused element with ephemeral index"""
    index: int
    description: str
    confidence: float
    created_at: float
    selected_indices: List[int]  # Store all selected indices for highlighting
    screenshot_path: Optional[str] = None


class FocusManager:
    """Focus manager using hybrid vision + DOM approach"""
    
    def __init__(self, page: Page, page_utils: PageUtils, deduper: Optional[InteractionDeduper] = None):
        self.page = page
        self.page_utils = page_utils
        self.overlay_manager = OverlayManager(page)
        self.deduper = deduper or InteractionDeduper()

        # Focus state
        self.focus_stack: List[FocusedElement] = []
        self.current_elements: List[Dict[str, Any]] = []
        self.current_overlay_data: List[Dict[str, Any]] = []
        self.current_screenshot: Optional[bytes] = None
        
        # Short-term memory for avoiding recent failures
        self.recent_failures: List[Dict[str, Any]] = []
        self.max_failure_memory = 10
        
        # Duplicate rejection tracking
        self.duplicate_rejection_count: int = 0
        self.duplicate_rejection_threshold: int = 2
        
        # Screenshot directory
        self.screenshot_dir = "focus_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def focus_on_elements(self, intent: str, page_info, overlay_manager: OverlayManager = None) -> bool:
        """
        Focus on elements based on natural language intent using AI-first approach.
        
        Args:
            intent: Natural language description of what to focus on
            page_info: Page information
            overlay_manager: Overlay manager instance
        
        Returns:
            True if focus was successful, False otherwise
        """
        try:
            print(f"🎯 Focus: {intent}")
            
            # Parse dedup settings from intent
            self._parse_dedup_settings(intent)

            try:
                action_intent = parse_action_intent(intent)
            except Exception:
                action_intent = None
            action_keyword = self._extract_action_keyword(intent, action_intent)
            self.deduper.set_action_keyword(action_keyword)
            
            # Step 1: Get all visible, interactable elements with ephemeral indices
            self.current_elements = self._get_visible_elements(page_info)
            if not self.current_elements:
                print("❌ No visible elements found")
                return False
            
            # Step 2: Filter out interacted elements if dedup is enabled
            if self.deduper.dedup_enabled:
                filtered_elements = self.deduper.filter_interacted_elements(self.current_elements, action_keyword)
                print(f"🔢 Found {len(filtered_elements)} visible elements (after dedup filtering)")
                self.current_elements = filtered_elements
            else:
                print(f"🔢 Found {len(self.current_elements)} visible elements")
            
            if not self.current_elements:
                print("❌ No elements available after deduplication")
                return False
            
            # Step 3: Take screenshot for AI vision
            self.current_screenshot = self.page.screenshot(type="png", full_page=False)

            # Step 4: AI selects indices based on intent + screenshot + elements
            selected_indices = self._select_indices_with_vision(intent, page_info, action_intent=action_intent)
            if not selected_indices:
                selected_indices = self._ai_select_elements(intent, self.current_elements, self.current_screenshot)

            if not selected_indices:
                print("❌ AI returned no-focus (low confidence)")
                return False
            
            # Step 4: Create focused element
            focused_element = self._create_focused_element(selected_indices, intent)
            if not focused_element:
                return False
            
            # Step 5: Add to focus stack
            self.focus_stack.append(focused_element)
            overlay_manager.remove_overlays()
            # Step 6: Capture screenshot with highlights
            self._capture_focus_screenshot(focused_element)
            
            print(f"✅ Focused on {len(selected_indices)} elements: {intent}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error in focus: {e}")
            return False
    
    def _get_visible_elements(self, page_info) -> List[Dict[str, Any]]:
        """Get all visible, interactable elements with ephemeral indices"""
        try:
            # Use overlay manager to get all visible elements
            overlay_data = self.overlay_manager.create_numbered_overlays(page_info, mode="all")
            self.current_overlay_data = overlay_data or []

            # Convert to our format with ephemeral indices
            elements = []
            for i, elem in enumerate(overlay_data):
                # Convert normalizedCoords to rect format
                if 'normalizedCoords' in elem and len(elem['normalizedCoords']) >= 4:
                    coords = elem['normalizedCoords']
                    # Convert from normalized coords back to pixel coords
                    viewport = self.page.viewport_size
                    rect = {
                        'x': (coords[1] / 1000) * viewport['width'],      # x_min
                        'y': (coords[0] / 1000) * viewport['height'],     # y_min
                        'width': ((coords[3] / 1000) * viewport['width']) - ((coords[1] / 1000) * viewport['width']),   # x_max - x_min
                        'height': ((coords[2] / 1000) * viewport['height']) - ((coords[0] / 1000) * viewport['height'])  # y_max - y_min
                    }
                else:
                    rect = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                
                # Create ephemeral index (0..N)
                element_data = {
                    'index': i,  # Ephemeral index
                    'overlay_index': elem.get('index'),  # Original overlay index
                    'description': elem.get('description', ''),
                    'tagName': elem.get('tagName', ''),
                    'text': elem.get('textContent', ''),
                    'role': elem.get('role', ''),
                    'className': elem.get('className', ''),
                    'id': elem.get('id', ''),
                    'rect': rect,
                    'normalizedCoords': elem.get('normalizedCoords', []),
                    'isClickable': elem.get('isClickable', False),
                    'isVisible': elem.get('isVisible', True),
                    'isEnabled': elem.get('isEnabled', True)
                }
                
                # Only include visible, interactable elements
                if element_data['isVisible'] and element_data['isEnabled']:
                    elements.append(element_data)
            
            return elements
            
        except Exception as e:
            traceback.print_exc()
            print(f"⚠️ Error getting visible elements: {e}")
            return []
    
    def _select_indices_with_vision(self, intent: str, page_info, action_intent: Optional[Any] = None) -> List[int]:
        """Use vision-first resolvers to rank overlays before invoking the LLM."""
        try:
            if not self.current_overlay_data or not self.current_elements:
                return []

            parsed_intent = action_intent
            if parsed_intent is None:
                try:
                    parsed_intent = parse_action_intent(intent)
                except Exception:
                    parsed_intent = None

            intent_lower = intent.lower()
            resolver = resolve_click_from_overlays
            mode = "click"

            if parsed_intent and parsed_intent.action == "type":
                resolver = resolve_field_from_overlays
                mode = "field"
            elif parsed_intent and parsed_intent.action == "select":
                resolver = resolve_select_from_overlays
                mode = "select"
            elif parsed_intent and parsed_intent.action == "datetime":
                resolver = resolve_datetime_from_overlays
                mode = "datetime"
            elif parsed_intent and parsed_intent.action == "upload":
                resolver = resolve_upload_from_overlays
                mode = "upload"
            elif any(word in intent_lower for word in ("type", "fill", "enter", "input")):
                resolver = resolve_field_from_overlays
                mode = "field"
            elif any(word in intent_lower for word in ("select", "dropdown", "combo", "option")):
                resolver = resolve_select_from_overlays
                mode = "select"
            elif any(word in intent_lower for word in ("date", "time", "schedule", "calendar")):
                resolver = resolve_datetime_from_overlays
                mode = "datetime"
            elif any(word in intent_lower for word in ("upload", "attach", "resume", "file")):
                resolver = resolve_upload_from_overlays
                mode = "upload"

            allowed_indices = {
                element.get('overlay_index')
                for element in self.current_elements
                if element.get('overlay_index') is not None
            }
            filtered_overlay_data = [
                o for o in self.current_overlay_data
                if o and o.get('index') in allowed_indices
            ]

            query_text = intent
            if action_intent and action_intent.target_text:
                query_text = action_intent.target_text

            resolution = resolver(query_text, filtered_overlay_data)
            scored = resolution.scored
            if not scored:
                return []

            top_score = scored[0][1]
            if top_score < 6:
                print(f"🤖 Vision-{mode} score too low ({top_score:.2f}), falling back to LLM")
                return []

            overlay_to_ephemeral = {}
            for idx, element in enumerate(self.current_elements):
                oidx = element.get('overlay_index')
                if oidx is not None:
                    overlay_to_ephemeral[oidx] = idx

            candidates: List[int] = []
            for oidx, score in scored:
                mapped = overlay_to_ephemeral.get(oidx)
                if mapped is None:
                    continue
                if score < max(2, top_score - 3):
                    continue
                if mapped not in candidates:
                    candidates.append(mapped)
                if len(candidates) >= 3:
                    break

            selected_overlay = resolution.best_index
            ordinal = parsed_intent.ordinal() if parsed_intent else None
            if ordinal is not None and scored:
                if ordinal == -1:
                    selected_overlay = scored[-1][0]
                elif 0 <= ordinal < len(scored):
                    selected_overlay = scored[ordinal][0]

            if selected_overlay is not None and selected_overlay in overlay_to_ephemeral:
                mapped_top = overlay_to_ephemeral[selected_overlay]
                if mapped_top in candidates:
                    candidates.remove(mapped_top)
                candidates.insert(0, mapped_top)

            if candidates:
                print(f"🎯 Vision-{mode} selected indices: {candidates}")
            return candidates
        except Exception as e:
            print(f"⚠️ Vision selection error: {e}")
            return []

    def _ai_select_elements(self, intent: str, elements: List[Dict[str, Any]], screenshot: bytes) -> List[int]:
        """
        AI selects element indices based on intent + screenshot + elements.
        Returns empty list if confidence is low.
        """
        try:
            # Create element summary for AI
            element_summary = self._create_element_summary(elements)
            
            # Check recent failures to avoid repeating them
            failure_context = self._get_failure_context(intent)
            
            prompt = f"""
            You are an expert at selecting UI elements based on user intent.

            USER INTENT: "{intent}"

            CURRENT ELEMENTS (with ephemeral indices 0-{len(elements)-1}):
            {chr(10).join(element_summary)}

            RECENT FAILURES TO AVOID:
            {failure_context}

            TASK: Select the index(es) that best satisfy the user intent.

            RANKING CRITERIA (in order of importance):
            1. Semantic match - does the element's text/description match the intent?
            2. Role/affordance - does the element's role match what the user wants to do?
            3. Proximity/association - is the element near other relevant elements?
            4. Visual cues - is the element visually prominent or styled appropriately?

            TIE-BREAKING (in order):
            1. Higher z-index (more likely to be interactive)
            2. Larger size (more prominent)
            3. Reading order (left-to-right, top-to-bottom)

            INSTRUCTIONS:
            - Return ONLY a JSON array of integers (ephemeral indices 0-{len(elements)-1})
            - If confidence is low, return an empty array []
            - For plural intents, return all matching indices
            - For singular intents, return the single best match
            - Avoid elements from recent failures
            - Be precise: better to return nothing than wrong elements

            OUTPUT FORMAT: [0, 3, 7] or [] for no-focus
            """

            response = generate_text(
                prompt=prompt,
                reasoning_level="medium",
                system_prompt="You are a UI element selection expert. Select element indices that best match the user's intent. Return only a JSON array of integers or empty array for no-focus.",
                model="gpt-5-mini",
                image=screenshot
            )
            
            # Parse AI response
            try:
                import json
                selected_indices = json.loads(response.strip())
                if not isinstance(selected_indices, list):
                    selected_indices = []
                
                # Validate indices
                valid_indices = [i for i in selected_indices if isinstance(i, int) and 0 <= i < len(elements)]
                
                if not valid_indices:
                    print("🤖 AI returned no-focus (low confidence or invalid indices)")
                    return []
                
                print(f"🤖 AI selected indices: {valid_indices}")
                filtered_indices = self.deduper.filter_duplicate_text_matches(valid_indices, elements, intent)
                if filtered_indices:
                    self.duplicate_rejection_count = 0
                    return filtered_indices

                # All candidates were filtered out due to duplicate text
                self.duplicate_rejection_count += 1
                if self.duplicate_rejection_count >= self.duplicate_rejection_threshold:
                    print("⚠️ Repeated duplicate selections detected, scrolling to break loop")
                    try:
                        if self.page_utils:
                            self.page_utils.scroll_page()
                    except Exception as scroll_error:
                        print(f"⚠️ Failed to scroll during duplicate loop break: {scroll_error}")
                    finally:
                        self.duplicate_rejection_count = 0
                return []
                
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse AI response: {response}")
                return []
                
        except Exception as e:
            print(f"⚠️ Error in AI selection: {e}")
            return []
    
    def _create_element_summary(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Create concise element summary for AI"""
        summary = []
        for i, elem in enumerate(elements):
            parts = [f"{i}"]
            
            # Type and role
            tag = elem.get('tagName', '')
            role = elem.get('role', '')
            if role and role != tag:
                parts.append(f"{tag}({role})")
            else:
                parts.append(tag)
            
            # Text content (truncated)
            text = elem.get('text', '').strip()
            if text:
                parts.append(f'"{text[:30]}"')
            
            # Key attributes
            if elem.get('isClickable'):
                parts.append("clickable")
            if elem.get('className'):
                parts.append(f"class:{elem.get('className', '')[:20]}")
            
            # Size info
            rect = elem.get('rect', {})
            area = (rect.get('width', 0) * rect.get('height', 0))
            if area > 10000:
                parts.append(f"large({int(area/1000)}k)")

            if self.deduper.dedup_enabled:
                center = self.deduper.extract_center_point(elem)
                if center:
                    if center.get('reference') == 'normalized':
                        parts.append(f"pos_norm=({center['x']:.1f},{center['y']:.1f})")
                    else:
                        parts.append(f"pos_px=({int(center['x'])},{int(center['y'])})")

            summary.append(" ".join(parts))
        
        return summary
    
    def _get_failure_context(self, intent: str) -> str:
        """Get context about recent failures to avoid repeating them"""
        if not self.recent_failures:
            return "None"
        
        # Get recent failures for similar intents
        similar_failures = [
            f for f in self.recent_failures 
            if any(word in f.get('intent', '').lower() for word in intent.lower().split())
        ]
        
        if not similar_failures:
            return "None"
        
        return "\n".join([
            f"- {f.get('intent', '')}: tried indices {f.get('indices', [])}"
            for f in similar_failures[-3:]  # Last 3 similar failures
        ])
    
    def _create_focused_element(self, selected_indices: List[int], intent: str) -> Optional[FocusedElement]:
        """Create a focused element from selected indices"""
        try:
            if not selected_indices:
                return None
            
            # Calculate confidence based on selection quality
            confidence = self._calculate_confidence(selected_indices, intent)
            
            # Create description
            if len(selected_indices) == 1:
                elem = self.current_elements[selected_indices[0]]
                description = f"{elem.get('text', '')} ({elem.get('tagName', '')})"
            else:
                description = f"{len(selected_indices)} elements matching '{intent}'"
            
            return FocusedElement(
                index=selected_indices[0] if len(selected_indices) == 1 else -1,  # -1 for multiple
                description=description,
                confidence=confidence,
                created_at=time.time(),
                selected_indices=selected_indices
            )
            
        except Exception as e:
            print(f"⚠️ Error creating focused element: {e}")
            return None
    
    def _calculate_confidence(self, selected_indices: List[int], intent: str) -> float:
        """Calculate confidence score for selected elements"""
        try:
            if not selected_indices:
                return 0.0
            
            total_confidence = 0.0
            for idx in selected_indices:
                if idx >= len(self.current_elements):
                    continue
                
                elem = self.current_elements[idx]
                confidence = 0.0
                
                # Semantic match score
                intent_words = set(intent.lower().split())
                elem_text = set(elem.get('text', '').lower().split())
                if intent_words & elem_text:
                    confidence += 0.4
                
                # Role/affordance score
                role = elem.get('role', '').lower()
                if 'button' in intent.lower() and role == 'button':
                    confidence += 0.3
                elif 'link' in intent.lower() and role == 'link':
                    confidence += 0.3
                elif 'input' in intent.lower() and role in ['textbox', 'searchbox']:
                    confidence += 0.3
                
                # Visual prominence score
                if elem.get('isClickable'):
                    confidence += 0.2
                
                # Size score (larger = more prominent)
                rect = elem.get('rect', {})
                area = (rect.get('width', 0) * rect.get('height', 0))
                if area > 10000:
                    confidence += 0.1
                
                total_confidence += min(confidence, 1.0)
            
            return total_confidence / len(selected_indices)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _capture_focus_screenshot(self, focused_element: FocusedElement) -> None:
        """Capture screenshot with focus highlights"""
        try:
            print(f"🔍 Debug: Selected indices: {focused_element.selected_indices}")
            print(f"🔍 Debug: Current elements count: {len(self.current_elements)}")
            
            # Step 1: Add visual highlights for focused elements
            self._highlight_focused_elements(focused_element)
            
            # Step 2: Wait for highlights to render
            import time
            time.sleep(0.5)
            
            # Step 3: Take screenshot with highlights
            timestamp = int(focused_element.created_at)
            filename = f"focus_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            screenshot_path = os.path.join(self.screenshot_dir, filename)
            
            screenshot = self.page.screenshot(type="png", full_page=False)
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            
            focused_element.screenshot_path = screenshot_path
            print(f"📸 Focus screenshot: {screenshot_path}")
            
            # Step 4: Remove highlights
            self._remove_focus_highlights()
            
        except Exception as e:
            print(f"⚠️ Error capturing focus screenshot: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_all_overlays(self) -> None:
        """Remove all element overlays from the page"""
        try:
            print("🔍 Debug: Removing overlays...")
            result = self.page.evaluate("""
                () => {
                    // Remove all element overlay elements
                    const overlays = document.querySelectorAll('.element-overlay, .element-number, .element-border, [data-automation-overlay-index]');
                    console.log('Found overlays to remove:', overlays.length);
                    overlays.forEach(el => el.remove());
                    return overlays.length;
                }
            """)
            print(f"🔍 Debug: Removed {result} overlays")
        except Exception as e:
            print(f"⚠️ Error removing overlays: {e}")
    
    def _highlight_focused_elements(self, focused_element: FocusedElement) -> None:
        """Add visual highlights to focused elements"""
        try:
            print(f"🔍 Debug: Highlighting {len(focused_element.selected_indices)} elements")
            
            if not focused_element.selected_indices:
                print("🔍 Debug: No selected indices to highlight")
                return
            
            # Get the elements to highlight
            elements_to_highlight = []
            for idx in focused_element.selected_indices:
                if idx < len(self.current_elements):
                    elem = self.current_elements[idx]
                    rect = elem.get('rect', {})
                    print(f"🔍 Debug: Element {idx}: {elem.get('description', 'Unknown')}")
                    print(f"🔍 Debug: Rect data: {rect}")
                    print(f"🔍 Debug: Rect type: {type(rect)}")
                    print(f"🔍 Debug: Rect keys: {rect.keys() if isinstance(rect, dict) else 'Not a dict'}")
                    elements_to_highlight.append(elem)
            
            print(f"🔍 Debug: Found {len(elements_to_highlight)} elements to highlight")
            
            if not elements_to_highlight:
                print("🔍 Debug: No valid elements to highlight")
                return
            
            # Create highlighting script that works with coordinates
            highlight_script = """
            (elements) => {
                console.log('Highlighting script called with', elements.length, 'elements');
                let highlightedCount = 0;
                
                elements.forEach((elem, index) => {
                    const rect = elem.rect;
                    console.log('Element', index, 'rect:', rect);
                    console.log('Element', index, 'rect type:', typeof rect);
                    console.log('Element', index, 'rect keys:', Object.keys(rect || {}));
                    console.log('Element', index, 'rect.width:', rect?.width, 'rect.height:', rect?.height);
                    
                    if (rect && rect.width > 0 && rect.height > 0) {
                        // Create highlight overlay
                        const highlight = document.createElement('div');
                        highlight.className = 'focus-highlight';
                        highlight.style.position = 'absolute';
                        highlight.style.left = (rect.x + window.scrollX) + 'px';
                        highlight.style.top = (rect.y + window.scrollY) + 'px';
                        highlight.style.width = rect.width + 'px';
                        highlight.style.height = rect.height + 'px';
                        highlight.style.border = '3px solid #00ff00';
                        highlight.style.boxShadow = '0 0 10px rgba(0, 255, 0, 0.5)';
                        highlight.style.backgroundColor = 'rgba(0, 255, 0, 0.1)';
                        highlight.style.zIndex = '9999';
                        highlight.style.pointerEvents = 'none';
                        
                        // Add label
                        const label = document.createElement('div');
                        label.className = 'focus-label';
                        label.textContent = `FOCUSED ${index + 1}`;
                        label.style.position = 'absolute';
                        label.style.background = '#00ff00';
                        label.style.color = '#000';
                        label.style.padding = '2px 6px';
                        label.style.fontSize = '12px';
                        label.style.fontWeight = 'bold';
                        label.style.borderRadius = '3px';
                        label.style.zIndex = '10000';
                        label.style.left = (rect.x + window.scrollX) + 'px';
                        label.style.top = (rect.y + window.scrollY - 25) + 'px';
                        label.style.pointerEvents = 'none';
                        
                        document.body.appendChild(highlight);
                        document.body.appendChild(label);
                        highlightedCount++;
                        console.log('Added highlight for element', index);
                    } else {
                        console.log('Skipping element', index, 'invalid rect');
                    }
                });
                
                console.log('Highlighting script completed. Highlighted', highlightedCount, 'elements');
                return highlightedCount;
            }
            """
            
            # Execute highlighting
            print("🔍 Debug: Executing highlighting script...")
            highlighted_count = self.page.evaluate(highlight_script, elements_to_highlight)
            print(f"🔍 Debug: Successfully highlighted {highlighted_count} elements")
            
        except Exception as e:
            print(f"⚠️ Error highlighting elements: {e}")
    
    def _remove_focus_highlights(self) -> None:
        """Remove focus highlights"""
        try:
            self.page.evaluate("""
                () => {
                    // Remove all focus highlight elements
                    const highlights = document.querySelectorAll('.focus-highlight, .focus-border, .focus-label');
                    highlights.forEach(el => el.remove());
                }
            """)
        except Exception as e:
            print(f"⚠️ Error removing highlights: {e}")
    
    def get_current_focus_context(self) -> Optional[FocusedElement]:
        """Get current focus context"""
        return self.focus_stack[-1] if self.focus_stack else None
    
    def is_element_in_focus(self, element_id: str, element_data: Dict[str, Any] = None) -> bool:
        """
        Check if an element is within the current focus context.
        
        Args:
            element_id: ID of the element to check
            element_data: Element data for matching (optional)
        
        Returns:
            bool: True if element is in focus, False otherwise
        """
        current_focus = self.get_current_focus_context()
        if not current_focus:
            return True  # No focus means all elements are available
        
        # For multi-element focus, check if index is in selected indices
        try:
            element_index = int(element_id)
            return element_index in current_focus.selected_indices
        except (ValueError, TypeError):
            # If element_id is not a number, assume it's in focus
            return True
    
    def clear_focus(self) -> None:
        """Clear all focus states"""
        self.focus_stack.clear()
        print("🧹 Cleared all focus states")
    
    def subfocus_on_elements(self, intent: str, page_info, overlay_manager: OverlayManager = None) -> bool:
        """
        Sub-focus on elements within the current focus context.
        This is the same as focus_on_elements in the AI-first approach.
        """
        return self.focus_on_elements(intent, page_info, overlay_manager)
    
    def undo_focus(self) -> bool:
        """Undo the last focus operation"""
        if len(self.focus_stack) > 1:
            self.focus_stack.pop()
            print("↩️ Undone last focus")
            return True
        elif self.focus_stack:
            self.focus_stack.clear()
            print("🧹 Cleared all focus states")
            return True
        else:
            print("❌ No focus to undo")
            return False
    
    def clear_all_focus(self) -> None:
        """Clear all focus states"""
        self.focus_stack.clear()
        print("🧹 Cleared all focus states")
    
    # -------------------- Deduplication Methods --------------------
    
    def _parse_dedup_settings(self, intent: str) -> None:
        """Parse dedup settings from intent string"""
        intent_lower = intent.lower()
        
        # Look for "dedup: enable" or "dedup: disable" patterns
        if "dedup: enable" in intent_lower or "dedup:enabled" in intent_lower:
            self.deduper.set_dedup_enabled(True)
        elif "dedup: disable" in intent_lower or "dedup:disabled" in intent_lower:
            self.deduper.set_dedup_enabled(False)

    def _extract_action_keyword(self, intent: str, action_intent: Optional[Any] = None) -> str:
        """Infer the primary action keyword for dedup purposes."""

        try:
            if action_intent and getattr(action_intent, "action", None):
                return str(action_intent.action).strip().lower()
        except Exception:
            pass

        intent_lower = (intent or "").lower()

        keyword_map = [
            ("type", "type"),
            ("fill", "type"),
            ("enter", "type"),
            ("input", "type"),
            ("select", "select"),
            ("choose", "select"),
            ("pick", "select"),
            ("upload", "upload"),
            ("attach", "upload"),
            ("press", "press"),
            ("scroll", "scroll"),
            ("navigate", "navigate"),
            ("go to", "navigate"),
            ("open", "navigate"),
            ("focus", "focus"),
            ("tap", "click"),
            ("click", "click"),
        ]

        for marker, normalized in keyword_map:
            if marker in intent_lower:
                return normalized

        # Default to click so common interactions are deduplicated by default
        return "click"

