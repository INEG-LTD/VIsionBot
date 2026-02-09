"""
Visual feedback and highlighting for action confirmation.
"""
from typing import List

from models import PageInfo

def highlight_selector(executor, selector: str) -> bool:
    script = """
    (selector) => {
        const el = document.querySelector(selector);
        if (!el) return false;
        const rect = el.getBoundingClientRect();
        const overlayId = '__codex_confirm_highlight';
        let overlay = document.getElementById(overlayId);
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = overlayId;
            overlay.style.position = 'absolute';
            overlay.style.pointerEvents = 'none';
            overlay.style.border = '3px solid #00ffae';
            overlay.style.boxShadow = '0 0 12px #00ffae';
            overlay.style.background = 'rgba(0, 255, 174, 0.18)';
            overlay.style.zIndex = 2147483647;
            document.body.appendChild(overlay);
        }
        overlay.style.left = (rect.left + window.scrollX) + 'px';
        overlay.style.top = (rect.top + window.scrollY) + 'px';
        overlay.style.width = rect.width + 'px';
        overlay.style.height = rect.height + 'px';
        return true;
    }
    """
    try:
        return bool(executor.page.evaluate(script, selector))
    except Exception:
        return False


def highlight_overlay(executor, overlay_index: int) -> bool:
    selector = f'[data-automation-overlay-index="{overlay_index}"]'
    if highlight_selector(executor, selector):
        return True
    fallback_selector = f'[data-overlay-index="{overlay_index}"]'
    return highlight_selector(executor, fallback_selector)


def highlight_box(executor, box: List[int], page_info: PageInfo) -> bool:
    if not box or len(box) != 4 or not page_info:
        return False
    try:
        y_min, x_min, y_max, x_max = box
        width_px = page_info.width or page_info.ss_pixel_w or 0
        height_px = page_info.height or page_info.ss_pixel_h or 0
        if not width_px or not height_px:
            return False
        left = max(0, int(x_min / 1000.0 * width_px))
        right = max(0, int(x_max / 1000.0 * width_px))
        top = max(0, int(y_min / 1000.0 * height_px))
        bottom = max(0, int(y_max / 1000.0 * height_px))
        width = max(1, right - left)
        height = max(1, bottom - top)
        script = """
        ({ left, top, width, height }) => {
            const overlayId = '__codex_confirm_highlight';
            let overlay = document.getElementById(overlayId);
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = overlayId;
                overlay.style.position = 'absolute';
                overlay.style.pointerEvents = 'none';
                overlay.style.border = '3px solid #00ffae';
                overlay.style.boxShadow = '0 0 12px #00ffae';
                overlay.style.background = 'rgba(0, 255, 174, 0.18)';
                overlay.style.zIndex = 2147483647;
                document.body.appendChild(overlay);
            }
            overlay.style.left = (left + window.scrollX) + 'px';
            overlay.style.top = (top + window.scrollY) + 'px';
            overlay.style.width = width + 'px';
            overlay.style.height = height + 'px';
            return true;
        }
        """
        return bool(executor.page.evaluate(script, {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }))
    except Exception:
        return False


def highlight_point(executor, x: int, y: int) -> bool:
    script = """
    ({x, y}) => {
        if (typeof x !== 'number' || typeof y !== 'number') return false;
        const overlayId = '__codex_confirm_highlight';
        let overlay = document.getElementById(overlayId);
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = overlayId;
            overlay.style.position = 'absolute';
            overlay.style.pointerEvents = 'none';
            overlay.style.borderRadius = '18px';
            overlay.style.zIndex = 2147483647;
            overlay.style.boxShadow = '0 0 10px #00ffae';
            overlay.style.background = 'rgba(0, 255, 174, 0.25)';
            document.body.appendChild(overlay);
        }
        const size = 36;
        overlay.style.width = overlay.style.height = size + 'px';
        overlay.style.left = (x - size / 2) + 'px';
        overlay.style.top = (y - size / 2) + 'px';
        return true;
    }
    """
    try:
        return bool(executor.page.evaluate(script, {"x": x, "y": y}))
    except Exception:
        return False


def highlight_click_location(executor, x: int, y: int, duration_ms: int = 1000) -> bool:
    """
    Highlight the click location for debugging purposes.
    Shows a pulsing circle at the click coordinates.
    
    Args:
        executor: Executor instance with page access
        x: X coordinate to highlight
        y: Y coordinate to highlight
        duration_ms: How long to show the highlight (milliseconds)
    
    Returns:
        True if highlight was successfully shown, False otherwise
    """
    script = """
    ({x, y, durationMs}) => {
        if (typeof x !== 'number' || typeof y !== 'number') return false;
        
        const overlayId = '__codex_click_debug_highlight';
        let overlay = document.getElementById(overlayId);
        
        // Remove existing overlay if present
        if (overlay) {
            overlay.remove();
        }
        
        // Create style element for animation if it doesn't exist
        let styleEl = document.getElementById('__codex_click_debug_style');
        if (!styleEl) {
            styleEl = document.createElement('style');
            styleEl.id = '__codex_click_debug_style';
            styleEl.textContent = `
                @keyframes codex-click-pulse {
                    0% {
                        transform: scale(0.8);
                        opacity: 1;
                        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7);
                    }
                    50% {
                        transform: scale(1.2);
                        opacity: 0.8;
                        box-shadow: 0 0 0 20px rgba(255, 0, 0, 0);
                    }
                    100% {
                        transform: scale(1);
                        opacity: 0.6;
                        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0);
                    }
                }
            `;
            document.head.appendChild(styleEl);
        }
        
        // Create overlay element
        overlay = document.createElement('div');
        overlay.id = overlayId;
        overlay.style.position = 'fixed';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = 2147483647;
        overlay.style.borderRadius = '50%';
        overlay.style.border = '4px solid #ff0000';
        overlay.style.background = 'rgba(255, 0, 0, 0.3)';
        overlay.style.boxShadow = '0 0 20px rgba(255, 0, 0, 0.8)';
        overlay.style.animation = 'codex-click-pulse 0.6s ease-out infinite';
        
        const size = 40;
        overlay.style.width = size + 'px';
        overlay.style.height = size + 'px';
        overlay.style.left = (x - size / 2) + 'px';
        overlay.style.top = (y - size / 2) + 'px';
        
        document.body.appendChild(overlay);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (overlay && overlay.parentNode) {
                overlay.style.transition = 'opacity 0.3s ease-out';
                overlay.style.opacity = '0';
                setTimeout(() => {
                    if (overlay && overlay.parentNode) {
                        overlay.remove();
                    }
                }, 300);
            }
        }, durationMs);
        
        return true;
    }
    """
    try:
        return bool(executor.browser.page.evaluate(script, {"x": x, "y": y, "durationMs": duration_ms}))
    except Exception:
        return False


def clear_highlight(executor) -> None:
    script = """
    () => {
        const overlay = document.getElementById('__codex_confirm_highlight');
        if (overlay && overlay.remove) {
            overlay.remove();
        }
        // Also clear click debug highlight
        const clickOverlay = document.getElementById('__codex_click_debug_highlight');
        if (clickOverlay && clickOverlay.remove) {
            clickOverlay.remove();
        }
        return true;
    }
    """
    try:
        executor.page.evaluate(script)
    except Exception:
        pass
