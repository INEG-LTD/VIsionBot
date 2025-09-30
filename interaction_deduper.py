"""
Interaction Deduper - Handles deduplication of user interactions.

This class manages the tracking of interacted elements to avoid duplicate actions
during automation. It's separate from the focus functionality to allow for
cleaner separation of concerns.
"""

import time
import re
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Set


class InteractionDeduper:
    """Handles deduplication of user interactions"""
    
    def __init__(self):
        # Deduplication system - tracks interacted elements
        self.interacted_elements: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.interaction_history_limit: int = -1  # -1 = keep full history
        self.dedup_enabled: bool = True  # Default to enabled
        self.current_action_keyword: str = "unknown"
        self.duplicate_rejection_count: int = 0
        self.duplicate_rejection_threshold: int = 2
    
    def extract_center_point(self, element: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract a stable center point for an element (normalized preferred)."""

        coords = (
            element.get('normalizedCoords')
            or element.get('box2d')
            or element.get('box_2d')
            or element.get('box2D')
        )

        if isinstance(coords, (list, tuple)) and len(coords) >= 4:
            try:
                y_min, x_min, y_max, x_max = [float(coords[i]) for i in range(4)]
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                return {
                    'x': round(x_center, 2),
                    'y': round(y_center, 2),
                    'reference': 'normalized'
                }
            except (TypeError, ValueError):
                pass

        rect = element.get('rect')
        if isinstance(rect, dict):
            try:
                x = float(rect.get('x', 0.0))
                y = float(rect.get('y', 0.0))
                width = float(rect.get('width', 0.0))
                height = float(rect.get('height', 0.0))
                x_center = x + (width / 2.0)
                y_center = y + (height / 2.0)
                return {
                    'x': round(x_center, 1),
                    'y': round(y_center, 1),
                    'reference': 'pixel'
                }
            except (TypeError, ValueError):
                pass

        return None

    def _normalize_action(self, action: Optional[str]) -> str:
        raw = (action or "").strip().lower()
        return raw or "unknown"

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # Collapse whitespace but preserve original casing so case-sensitive comparisons remain valid
        collapsed = re.sub(r"\s+", " ", str(text)).strip()
        return collapsed

    def _generate_element_signature(self, element: Dict[str, Any], action: Optional[str] = None) -> str:
        """Generate a stable signature using action + text content only."""

        try:
            action_key = self._normalize_action(action or element.get('interaction_type') or element.get('action'))
            text_source = (
                element.get('text')
                or element.get('textContent')
                or element.get('description')
                or element.get('element_label')
                or element.get('ariaLabel')
                or element.get('aria_label')
            )
            normalized_text = self._normalize_text(text_source)
            if not normalized_text:
                normalized_text = "(no-text)"

            raw_sig = f"{action_key}|{normalized_text}"
            return self._stable_hash(raw_sig)
        except Exception:
            try:
                return self._stable_hash(str(element))
            except Exception:
                return ""

    def _extract_visible_text(self, element: Dict[str, Any]) -> str:
        """Extract the most relevant visible text from an element for dedup comparison."""

        if not isinstance(element, dict):
            return ""

        for key in ('text', 'textContent', 'description', 'element_label', 'ariaLabel', 'aria_label'):
            value = element.get(key)
            if isinstance(value, str) and value.strip():
                return self._normalize_text(value)
        return ""

    def _collect_dedup_texts(self) -> Set[str]:
        """Return set of normalized texts that have already been interacted with."""
        texts: Set[str] = set()
        for entry in self.interacted_elements.values():
            element_snapshot = entry.get('element') or {}
            text = self._extract_visible_text(element_snapshot)
            if text:
                texts.add(text)
        return texts

    def _stable_hash(self, text: str) -> str:
        """Generate a stable hash for text"""
        import hashlib
        try:
            return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        except Exception:
            return text or ""

    def filter_interacted_elements(self, elements: List[Dict[str, Any]], action: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter out elements that have been interacted with"""
        filtered = []
        filtered_count = 0
        action_key = (action or self.current_action_keyword or "unknown").strip().lower()
        
        for element in elements:
            signature = self._generate_element_signature(element, action_key)
            if signature and signature in self.interacted_elements:
                filtered_count += 1
                print(f"🚫 Filtered out interacted element for action '{action_key}': {element.get('text', '')[:30]}...")
            else:
                filtered.append(element)
        
        if filtered_count > 0:
            print(f"🧹 Filtered out {filtered_count} previously interacted elements")
        
        return filtered

    def filter_duplicate_text_matches(
        self,
        indices: List[int],
        elements: List[Dict[str, Any]],
        intent: str,
    ) -> List[int]:
        """Remove indices whose text matches prior interactions when dedup is enabled."""

        if not self.dedup_enabled or not self.interacted_elements:
            return indices

        dedup_texts = self._collect_dedup_texts()
        if not dedup_texts:
            return indices

        filtered: List[int] = []
        duplicates: List[Dict[str, Any]] = []

        for idx in indices:
            if idx >= len(elements):
                continue

            element = elements[idx]
            text = self._extract_visible_text(element)
            if text and text in dedup_texts:
                duplicates.append({'index': idx, 'text': text})
                print(f"🚫 Duplicate text match for index {idx}: '{text}'")
                continue

            filtered.append(idx)

        if duplicates:
            self._record_duplicate_selection_failure(intent, duplicates)

        return filtered

    def _record_duplicate_selection_failure(self, intent: str, duplicates: List[Dict[str, Any]]) -> None:
        """Record duplicate selections to steer future attempts."""

        if not duplicates:
            return

        # Note: This would need to be passed to the focus manager if we want to track failures
        # For now, we'll just log it
        print(f"📝 Recorded duplicate selection failure for intent: {intent}")

    def mark_element_as_interacted(self, element: Dict[str, Any], interaction_type: str = "click") -> None:
        """Mark an element as interacted with for deduplication"""

        if not element:
            print("❌ No element to mark as interacted with")
            return

        signature = self._generate_element_signature(element, interaction_type)
        if not signature:
            print("❌ No signature for element")
            return None

        element_snapshot = {
            'tagName': element.get('tagName') or element.get('tag_name') or element.get('element_type'),
            'text': element.get('text') or element.get('textContent') or element.get('description'),
            'description': element.get('description') or element.get('element_label'),
            'href': element.get('href'),
            'ariaLabel': element.get('ariaLabel') or element.get('aria_label'),
            'id': element.get('id'),
            'role': element.get('role') or element.get('element_type'),
            'overlayIndex': element.get('overlayIndex') or element.get('overlay_index'),
            'box2d': element.get('box2d') or element.get('box_2d') or element.get('normalizedCoords'),
            'normalizedCoords': element.get('normalizedCoords') or element.get('box2d') or element.get('box_2d'),
        }

        center = self.extract_center_point({**element_snapshot, **element})
        if center:
            element_snapshot['position'] = center

        record = {
            'signature': signature,
            'element': element_snapshot,
            'interaction_type': interaction_type,
            'timestamp': time.time(),
            'position': center,
        }

        # Refresh recency ordering
        if signature in self.interacted_elements:
            self.interacted_elements.pop(signature, None)
        self.interacted_elements[signature] = record
        self._enforce_interaction_limit()

        if self.dedup_enabled:
            text_preview = (element_snapshot.get('text') or element_snapshot.get('description') or '')[:30]
            print(f"📝 Marked element as {interaction_type}ed: {text_preview}...")

    def clear_interacted_elements(self) -> None:
        """Clear all interacted elements from dedup tracking"""
        self.interacted_elements.clear()
        print("🧹 Cleared all interacted elements from dedup tracking")

    def get_interacted_elements_count(self) -> int:
        """Get count of currently tracked interacted elements"""
        return len(self.interacted_elements)

    def get_interacted_element_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a copy of the recent interacted element history."""

        if limit is None:
            limit = self.interaction_history_limit

        records = list(self.interacted_elements.values())
        if limit is not None and limit >= 0:
            records = records[-limit:]

        # Return shallow copies to avoid accidental mutation
        history: List[Dict[str, Any]] = []
        for entry in records:
            element_copy = dict(entry.get('element') or {})
            history.append({
                'signature': entry.get('signature'),
                'element': element_copy,
                'interaction_type': entry.get('interaction_type'),
                'timestamp': entry.get('timestamp'),
                'position': entry.get('position'),
            })
        return history

    def set_interaction_history_limit(self, limit: int) -> None:
        """Configure how many interacted elements to retain (-1 = unlimited)."""
        self.interaction_history_limit = max(-1, int(limit))
        self._enforce_interaction_limit()

    def _enforce_interaction_limit(self) -> None:
        if self.interaction_history_limit is None or self.interaction_history_limit < 0:
            return
        while len(self.interacted_elements) > self.interaction_history_limit:
            self.interacted_elements.popitem(last=False)

    def set_dedup_enabled(self, enabled: bool) -> None:
        """Enable or disable deduplication"""
        self.dedup_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🧹 Deduplication {status}")

    def set_action_keyword(self, action_keyword: str) -> None:
        """Set the current action keyword for deduplication context"""
        self.current_action_keyword = action_keyword
