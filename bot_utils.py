import asyncio
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
import uuid
import os
import time
from rich import print as rprint

debug_mode = True

def dprint(text: str):
    if debug_mode:
        print(text)


def start_browser(browser_position: str = "right") -> tuple[Browser, BrowserContext, Page]:
    # Setup window layout first
    try:
        from window_adjustment import stage_manager, visible_work_area_top_left, split_rect, move_frontmost_window
        
        with stage_manager(disabled=True):
            work_area = visible_work_area_top_left()
            browser_rect, terminal_rect = split_rect(work_area, browser_position, 0.4, 8)
            
            # Move terminal to left side
            move_frontmost_window(terminal_rect)
            time.sleep(0.15)
            
            dprint("✅ Window layout ready - terminal left, browser space right")
            
            # Your existing browser launch code stays exactly the same
            playwright = None
            browser = None
            context = None
            page = None
            
            try:
                # Ensure we're not in an asyncio context
                try:
                    asyncio.get_running_loop()
                    print("⚠️ Warning: Running in asyncio context, this may cause issues")
                except RuntimeError:
                    pass  # No asyncio loop running, which is good
                    
                playwright = sync_playwright().start()
                
                try:
                    unique_id = str(uuid.uuid4())[:8]
                    chrome_user_data_dir = os.path.expanduser(f"~/Library/Application Support/Google/Chrome/Automation_{unique_id}")
                    
                    # Create automation profile directory if it doesn't exist
                    os.makedirs(chrome_user_data_dir, exist_ok=True)
                    
                    # Launch with window position arguments from the window management API
                    
                    x, y, w, h = map(int, browser_rect)
                    browser = playwright.chromium.launch_persistent_context(
                        viewport={"width": w, "height": h},
                        user_data_dir=chrome_user_data_dir,
                        headless=False,
                        args=[
                            "--no-sandbox", 
                            "--disable-dev-shm-usage", 
                            "--disable-blink-features=AutomationControlled",
                            '--disable-features=VizDisplayCompositor',
                            f"--window-position={x},{y}",
                            f"--window-size={w},{h}",
                            ],
                        channel="chrome"
                    )
                    
                    pages = browser.pages
                    if pages:
                        page = pages[0]
                    else:
                        page = browser.new_page()
                    context = page.context
                        
                    dprint("✅ Browser started")
                        
                    return browser, context, page
                
                except Exception as e:
                    print(f"❌ Error starting browser: {e}")
                    return None, None, None
            except Exception as e:
                print(f"❌ Error starting browser: {e}")
                return None, None, None
    except Exception as e:
        print(f"⚠️ Window setup failed, continuing normally: {e}")
    
    
if __name__ == "__main__":
    start_browser()