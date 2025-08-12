# macOS only
# pip install playwright pyobjc
# playwright install chromium   # or use channel="chrome" if Chrome is installed

import argparse
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

from AppKit import NSScreen
from playwright.sync_api import sync_playwright

# -------------------- Stage Manager toggle --------------------

def _read_bool(domain: str, key: str) -> bool:
    out = subprocess.run(
        ["defaults", "read", domain, key],
        capture_output=True, text=True
    ).stdout.strip().lower()
    return out in ("1", "true", "yes")

def stage_manager_enabled() -> bool:
    return _read_bool("com.apple.WindowManager", "GloballyEnabled")

def _set_stage_manager(enabled: bool, hard_reload: bool = False):
    subprocess.run(
        ["defaults", "write", "com.apple.WindowManager", "GloballyEnabled", "-bool", "true" if enabled else "false"],
        check=False
    )
    # Nudge Control Center so the toggle applies immediately.
    subprocess.run(["killall", "ControlCenter"], check=False)
    if hard_reload:
        # More disruptive; only if you absolutely need it.
        subprocess.run(["killall", "Dock"], check=False)

@contextmanager
def stage_manager_temporarily(disabled: bool = True, hard_reload: bool = False):
    prev = stage_manager_enabled()
    try:
        target = not disabled
        if prev != target:
            _set_stage_manager(target, hard_reload=hard_reload)
            time.sleep(0.25)
        yield
    finally:
        if stage_manager_enabled() != prev:
            _set_stage_manager(prev, hard_reload=hard_reload)
            time.sleep(0.25)

# -------------------- Geometry helpers (top-left origin) --------------------

def visible_work_area_top_left() -> Tuple[int, int, int, int]:
    """Main display visible frame in top-left origin coords."""
    scr = NSScreen.mainScreen()
    full = scr.frame()        # bottom-left origin
    vis  = scr.visibleFrame() # bottom-left origin; excludes Dock/menu bar

    full_w, full_h = int(full.size.width), int(full.size.height)
    vis_x_bl, vis_y_bl = int(vis.origin.x), int(vis.origin.y)
    vis_w, vis_h = int(vis.size.width), int(vis.size.height)

    vis_x_tl = vis_x_bl
    vis_y_tl = full_h - (vis_y_bl + vis_h)
    return (vis_x_tl, vis_y_tl, vis_w, vis_h)

def split_rect(work, side: str, frac: float, margin: int):
    x, y, w, h = work
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

# -------------------- Window move + Playwright launch --------------------

def move_frontmost_window(rect):
    x, y, w, h = map(int, rect)
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

def launch_chrome_at(pw, rect, channel="chrome"):
    x, y, w, h = map(int, rect)
    args = [f"--window-position={x},{y}", f"--window-size={w},{h}"]
    browser = pw.chromium.launch(channel=channel, headless=False, args=args)
    context = browser.new_context(viewport={"width": w, "height": h})
    page = context.new_page()
    return browser, context, page

# -------------------- Orchestrator --------------------

@dataclass
class Layout:
    side: str = "right"    # left|right|top|bottom
    fraction: float = 0.60 # 0.2..0.9
    margin: int = 8
    channel: str = "chrome"  # "chrome" or "chromium"

def run(layout: Layout, url: str, hard_reload=False):
    # Disable Stage Manager for the session, then restore.
    with stage_manager_temporarily(disabled=True, hard_reload=hard_reload):
        work = visible_work_area_top_left()
        browser_rect, term_rect = split_rect(work, layout.side, layout.fraction, layout.margin)

        # 1) Move terminal first so it stays visible
        move_frontmost_window(term_rect)
        time.sleep(0.15)

        # 2) Launch Chrome exactly where we want it
        with sync_playwright() as pw:
            browser, ctx, page = launch_chrome_at(pw, browser_rect, channel=layout.channel)
            page.goto(url)
            # TODO: your automation here
            page.wait_for_timeout(2000)
            # browser.close()  # keep open while debugging

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tile Terminal + Chrome with Stage Manager temporarily disabled.")
    ap.add_argument("--side", choices=["left", "right", "top", "bottom"], default="right")
    ap.add_argument("--frac", type=float, default=0.60, help="Browser share of screen (0.2..0.9).")
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--channel", choices=["chrome", "chromium"], default="chrome")
    ap.add_argument("--url", default="https://example.com")
    ap.add_argument("--hard-reload", action="store_true",
                    help="Also restart Dock (more disruptive) when toggling Stage Manager.")
    args = ap.parse_args()

    run(
        Layout(side=args.side, fraction=args.frac, margin=args.margin, channel=args.channel),
        url=args.url,
        hard_reload=args.hard_reload
    )