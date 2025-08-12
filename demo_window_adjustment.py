# macOS only
# pip install playwright pyobjc
# playwright install chromium   # or use channel="chrome" if Chrome is installed

import subprocess, json, time
from dataclasses import dataclass
from typing import Tuple, Optional

# ---- AppKit (pyobjc) to get real work area (excludes Dock & menu bar)
from AppKit import NSScreen
from playwright.sync_api import sync_playwright

@dataclass
class Layout:
    # "left" | "right" | "top" | "bottom"
    browser_side: str = "right"
    # fraction of the available work area given to the browser (0.2..0.9)
    browser_fraction: float = 0.58
    # pixel margin around the windows
    margin: int = 8
    # if Stage Manager is ON but we cannot read Apple's hidden inset, use this
    stage_left_inset_fallback: int = 80
    # force-disable Stage Manager adjustments (for testing)
    ignore_stage_manager: bool = False

# ---------- Stage Manager detection ----------

def _defaults_read_bool(domain: str, key: str) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["defaults", "read", domain, key],
            check=False, capture_output=True, text=True
        ).stdout.strip().lower()
        if out in ("1", "true", "yes"):
            return True
        if out in ("0", "false", "no"):
            return False
    except Exception:
        pass
    return None

def _defaults_read_int(domain: str, key: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["defaults", "read", domain, key],
            check=False, capture_output=True, text=True
        ).stdout.strip()
        return int(out)
    except Exception:
        return None

def is_stage_manager_enabled() -> bool:
    val = _defaults_read_bool("com.apple.WindowManager", "GloballyEnabled")
    return bool(val)

def read_stage_manager_left_inset() -> Optional[int]:
    # This key exists on some macOS builds; if missing, return None.
    # When present, 0 means "allow full width"; positive values leave a gap at the left.
    return _defaults_read_int("com.apple.WindowManager", "StageFrameMinimumHorizontalInset")

# ---------- Geometry helpers (top-left origin) ----------

def _screen_frames_top_origin() -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """
    Returns (screen_full, visible_work_area) as (x,y,w,h) using TOP-left origin.
    """
    scr = NSScreen.mainScreen()
    full = scr.frame()          # NSRect in bottom-left origin
    vis  = scr.visibleFrame()   # excludes Dock & menu bar

    # Convert to top-left origin
    full_w, full_h = int(full.size.width), int(full.size.height)
    full_top = (0, 0, full_w, full_h)

    # visibleFrame origin is bottom-left; convert to top-left
    vis_x_bl = int(vis.origin.x)
    vis_y_bl = int(vis.origin.y)
    vis_w    = int(vis.size.width)
    vis_h    = int(vis.size.height)
    vis_x_tl = vis_x_bl
    vis_y_tl = full_h - (vis_y_bl + vis_h)

    return full_top, (vis_x_tl, vis_y_tl, vis_w, vis_h)

def _apply_stage_manager_inset(work: Tuple[int,int,int,int], inset_left: int) -> Tuple[int,int,int,int]:
    x,y,w,h = work
    x += inset_left
    w -= inset_left
    return (x,y,w,h)

def _split_rect(work: Tuple[int,int,int,int], side: str, frac: float, margin: int):
    x,y,w,h = work
    frac = max(0.2, min(0.9, frac))
    m = margin

    if side in ("left", "right"):
        bw = int(w * frac) - 2*m
        tw = w - bw - 3*m
        if side == "left":
            browser = (x+m, y+m, bw, h-2*m)
            terminal = (x+2*m+bw, y+m, tw, h-2*m)
        else:
            terminal = (x+m, y+m, tw, h-2*m)
            browser  = (x+2*m+tw, y+m, bw, h-2*m)
    else:
        bh = int(h * frac) - 2*m
        th = h - bh - 3*m
        if side == "top":
            browser = (x+m, y+m, w-2*m, bh)
            terminal = (x+m, y+2*m+bh, w-2*m, th)
        else:
            terminal = (x+m, y+m, w-2*m, th)
            browser  = (x+m, y+2*m+th, w-2*m, bh)
    return browser, terminal

# ---------- Move terminal (AppleScript) & launch Chrome ----------

def move_frontmost_window(rect):
    x,y,w,h = map(int, rect)
    right, bottom = x + w, y + h
    ascript = f'''
    set targetBounds to {{{x}, {y}, {right}, {bottom}}}
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    if frontApp is "Terminal" then
        tell application "Terminal" to set bounds of front window to targetBounds
    else if frontApp is "iTerm2" then
        tell application "iTerm2" to set bounds of front window to targetBounds
    else if frontApp is "Visual Studio Code" then
        tell application "System Events" to tell process "Visual Studio Code"
            set position of front window to {{{x}, {y}}}
            set size of front window to {{{w}, {h}}}
        end tell
    else
        tell application "System Events" to tell process frontApp
            try
                set position of front window to {{{x}, {y}}}
                set size of front window to {{{w}, {h}}}
            end try
        end tell
    end if
    '''
    subprocess.run(["osascript", "-e", ascript], check=False)

def launch_chrome_at(pw, rect):
    x,y,w,h = map(int, rect)
    args = [f"--window-position={x},{y}", f"--window-size={w},{h}"]
    browser = pw.chromium.launch(channel="chrome", headless=False, args=args)  # use Chrome.app
    context = browser.new_context(viewport={"width": w, "height": h})
    page = context.new_page()
    return browser, context, page

# ---------- Public API ----------

def place_terminal_and_browser(layout: Layout) -> None:
    _, work = _screen_frames_top_origin()

    sm_enabled = is_stage_manager_enabled() if not layout.ignore_stage_manager else False

    # Try to read Apple's left inset when Stage Manager is on; otherwise use fallback.
    if sm_enabled:
        inset = read_stage_manager_left_inset()
        inset_left = inset if (inset is not None and inset >= 0) else layout.stage_left_inset_fallback
        work = _apply_stage_manager_inset(work, inset_left)

    browser_rect, term_rect = _split_rect(work, layout.browser_side, layout.browser_fraction, layout.margin)

    # Move terminal first so it doesn't get covered
    move_frontmost_window(term_rect)
    time.sleep(0.15)

    with sync_playwright() as pw:
        browser, ctx, page = launch_chrome_at(pw, browser_rect)
        page.goto("https://example.com")
        # ... your automation here ...
        page.wait_for_timeout(2000)
        # browser.close()  # keep open while debugging

if __name__ == "__main__":
    place_terminal_and_browser(Layout(browser_side="right", browser_fraction=0.6))