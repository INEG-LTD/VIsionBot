from playwright.sync_api import Page, sync_playwright
from job_pref import JobPreferences
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import random
from input_handling import get_html_form_inputs, get_html_button_inputs

def visit_site(site_url: str) -> tuple[str, Page]:
    proxy_server = "http://proxy-eu.proxy-cheap.com:5959"
    proxy_username = "pcTQGZIWdI-res_sc-uk_england"
    proxy_password = "PC_8cSamU8Txcin2Ooik"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            proxy={
                "server": proxy_server,
                "username": proxy_username,
                "password": proxy_password
            }
        )
        
        page = browser.new_page()
        
        stealth_sync(page)
        
        page.goto(site_url)
        
        # get the page source
        page_source = page.content()
        
        # take a screenshot
        page.screenshot(path=f"{site_url}.png")

        return page_source, page

if __name__ == "__main__":
    page_source, page = visit_site("https://www.monster.co.uk/")
    print(page_source)