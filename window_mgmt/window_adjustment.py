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

# ---------- Stage Manager (temporary toggle) ----------

def _read_bool(domain: str, key: str) -> bool:
    out = subprocess.run(["defaults", "read", domain, key],
                         capture_output=True, text=True).stdout.strip().lower()
    return out in ("1", "true", "yes")

def _write_bool(domain: str, key: str, value: bool):
    subprocess.run(["defaults", "write", domain, key, "-bool", "true" if value else "false"], check=False)

def _nudge_ui(hard_reload: bool):
    subprocess.run(["killall", "ControlCenter"], check=False)
    if hard_reload:
        subprocess.run(["killall", "Dock"], check=False)

@contextmanager
def stage_manager(disabled: bool = True, hard_reload: bool = False):
    """Disable Stage Manager inside the 'with' block; restore afterwards."""
    domain, key = "com.apple.WindowManager", "GloballyEnabled"
    before = _read_bool(domain, key)
    target = not disabled
    try:
        if before != target:
            _write_bool(domain, key, target)
            _nudge_ui(hard_reload)
            time.sleep(0.25)
        yield
    finally:
        if _read_bool(domain, key) != before:
            _write_bool(domain, key, before)
            _nudge_ui(hard_reload)
            time.sleep(0.25)

# ---------- Geometry (top-left origin) ----------

def visible_work_area_top_left() -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) for the main screen's visible frame in top-left origin."""
    scr = NSScreen.mainScreen()
    full = scr.frame()         # bottom-left origin
    vis  = scr.visibleFrame()  # bottom-left origin (excludes Dock/Menu Bar)

    full_h = int(full.size.height)
    vis_x_bl, vis_y_bl = int(vis.origin.x), int(vis.origin.y)
    vis_w, vis_h = int(vis.size.width), int(vis.size.height)

    vis_x_tl = vis_x_bl
    vis_y_tl = full_h - (vis_y_bl + vis_h)
    return (vis_x_tl, vis_y_tl, vis_w, vis_h)

def split_rect(work, side: str, frac: float, margin: int):
    """Return (browser_rect, terminal_rect) with simple clamped fraction and margins."""
    x, y, w, h = work
    f = max(0.2, min(0.9, frac))
    m = margin

    if side in ("left", "right"):
        bw = int(w * f) - 2*m
        tw = w - bw - 3*m
        b = (x+m, y+m, bw, h-2*m)
        t = (x+2*m+bw, y+m, tw, h-2*m)
        return (b, t) if side == "left" else ( (x+2*m+tw, y+m, bw, h-2*m), (x+m, y+m, tw, h-2*m) )
    else:
        bh = int(h * f) - 2*m
        th = h - bh - 3*m
        b = (x+m, y+m, w-2*m, bh)
        t = (x+m, y+2*m+bh, w-2*m, th)
        return (b, t) if side == "top" else ( (x+m, y+2*m+th, w-2*m, bh), (x+m, y+m, w-2*m, th) )

# ---------- Window move + Playwright ----------

def move_frontmost_window(rect):
    """Best-effort move/resize the current frontmost app window via AppleScript."""
    x, y, w, h = map(int, rect)
    right, bottom = x + w, y + h

    ascript = f'''
    set targetBounds to {{{x}, {y}, {right}, {bottom}}}
    tell application "System Events"
        set frontProc to first application process whose frontmost is true
        try
            tell frontProc
                set position of front window to {{{x}, {y}}}
                set size of front window to {{{w}, {h}}}
            end tell
        on error
            set appName to name of frontProc
            if appName is "Terminal" then
                tell application "Terminal" to set bounds of front window to targetBounds
            else if appName is "iTerm2" then
                tell application "iTerm2" to set bounds of front window to targetBounds
            else if appName is "Visual Studio Code" then
                tell application "System Events" to tell process "Visual Studio Code"
                    set position of front window to {{{x}, {y}}}
                    set size of front window to {{{w}, {h}}}
                end tell
            end if
        end try
    end tell
    '''
    subprocess.run(["osascript", "-e", ascript], check=False)

def launch_chrome_at(pw, rect, channel="chrome"):
    x, y, w, h = map(int, rect)
    args = [f"--window-position={x},{y}", f"--window-size={w},{h}"]
    browser = pw.chromium.launch(channel=channel, headless=False, args=args)
    ctx = browser.new_context(viewport={"width": w, "height": h})
    page = ctx.new_page()
    return browser, ctx, page

# ---------- Orchestration ----------

@dataclass
class Layout:
    side: str = "right"     # left|right|top|bottom
    fraction: float = 0.60  # 0.2..0.9
    margin: int = 8
    channel: str = "chrome" # "chrome" or "chromium"

def run(layout: Layout, url: str, hard_reload=False):
    with stage_manager(disabled=True, hard_reload=hard_reload):
        work = visible_work_area_top_left()
        browser_rect, term_rect = split_rect(work, layout.side, layout.fraction, layout.margin)

        # Move terminal first so it stays visible
        move_frontmost_window(term_rect)
        time.sleep(0.15)

        # Launch Chrome at the exact rectangle
        with sync_playwright() as pw:
            browser, ctx, page = launch_chrome_at(pw, browser_rect, channel=layout.channel)
            page.goto(url)
            page.wait_for_timeout(2000)
            # browser.close()  # keep open while debugging

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tile Terminal + Chrome with Stage Manager temporarily disabled.")
    ap.add_argument("--side", choices=["left", "right", "top", "bottom"], default="right")
    ap.add_argument("--frac", type=float, default=0.60, help="Browser share of screen (0.2..0.9).")
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--channel", choices=["chrome", "chromium"], default="chrome")
    ap.add_argument("--url", default="https://example.com")
    ap.add_argument("--hard-reload", action="store_true", help="Also restart Dock when toggling Stage Manager.")
    args = ap.parse_args()

    run(
        Layout(side=args.side, fraction=args.frac, margin=args.margin, channel=args.channel),
        url=args.url,
        hard_reload=args.hard_reload
    )