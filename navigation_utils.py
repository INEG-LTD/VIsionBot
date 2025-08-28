"""
Navigation utilities for the vision bot.
"""

from typing import Tuple, List, Dict, Any


class NavigationUtils:
    """Utility class for navigation operations."""
    
    @staticmethod
    def install_spa_hook_once() -> str:
        """JavaScript code to install SPA navigation hook once."""
        return """
        if (window._spa_nav_count === undefined) {
            window._spa_nav_count = 0;
            window._spa_nav_history = [];
            
            // Hook pushState/replaceState
            const originalPushState = history.pushState;
            const originalReplaceState = history.replaceState;
            
            history.pushState = function(...args) {
                window._spa_nav_count++;
                window._spa_nav_history.push({type: 'push', url: args[2], timestamp: Date.now()});
                return originalPushState.apply(this, args);
            };
            
            history.replaceState = function(...args) {
                window._spa_nav_count++;
                window._spa_nav_history.push({type: 'replace', url: args[2], timestamp: Date.now()});
                return originalReplaceState.apply(this, args);
            };
            
            // Hook popstate
            window.addEventListener('popstate', function() {
                window._spa_nav_count++;
                window._spa_nav_history.push({type: 'popstate', url: location.href, timestamp: Date.now()});
            });
            
            // Hook hashchange
            window.addEventListener('hashchange', function() {
                window._spa_nav_count++;
                window._spa_nav_history.push({type: 'hashchange', url: location.href, timestamp: Date.now()});
            });
        }
        """
    
    @staticmethod
    def spa_nav_count() -> str:
        """JavaScript code to get SPA navigation count."""
        return "window._spa_nav_count || 0"
    
    @staticmethod
    def get_document_dimensions() -> str:
        """JavaScript code to get document dimensions."""
        return """
        () => {
            const doc = document.documentElement;
            const body = document.body;
            return [
                Math.max(doc.scrollWidth, body.scrollWidth, doc.offsetWidth, body.offsetWidth, doc.clientWidth),
                Math.max(doc.scrollHeight, body.scrollHeight, doc.offsetHeight, body.offsetHeight, doc.clientHeight)
            ];
        }
        """
    
    @staticmethod
    def flatten_ignore(elem_map: Dict[int, List]) -> List[Dict]:
        """Flatten element map for processing."""
        result = []
        for idx, items in elem_map.items():
            for item in items:
                result.append({"index": idx, **item})
        return result
    
    @staticmethod
    def center_from_box(box: List[int], meta) -> Tuple[int, int]:
        """Calculate center coordinates from bounding box."""
        ymin, xmin, ymax, xmax = box
        
        # Convert normalized coordinates to pixels
        from vision_bot import _denormalize_from_1000
        pixel_xmin = _denormalize_from_1000(xmin, meta.width)
        pixel_ymin = _denormalize_from_1000(ymin, meta.height)
        pixel_xmax = _denormalize_from_1000(xmax, meta.width)
        pixel_ymax = _denormalize_from_1000(ymax, meta.height)
        
        # Calculate center
        center_x = (pixel_xmin + pixel_xmax) // 2
        center_y = (pixel_ymin + pixel_ymax) // 2
        
        return center_x, center_y
    
    @staticmethod
    def is_click_target_clear(page, x: int, y: int) -> bool:
        """Check if the click target at coordinates is clear and clickable."""
        try:
            element = page.element_from_point(x, y)
            if not element:
                return False
            
            # Check if it's a clickable element
            tag = element.tag_name.lower()
            if tag in ['button', 'a', 'input']:
                return True
            
            # Check if it has click handlers
            has_click = element.get_attribute('onclick') or element.get_attribute('role') == 'button'
            if has_click:
                return True
            
            # Check if it's in a clickable container
            parent = element.query_selector('xpath=ancestor::button | ancestor::a | ancestor::[role="button"]')
            return parent is not None
            
        except Exception:
            return False
