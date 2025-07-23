from playwright.sync_api import sync_playwright
from job_pref import JobPreferences
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import random

def find_job_listings_with_playwright(preferences: JobPreferences):
        
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

        # Example: Visit a job board (e.g., Indeed, replace with actual site as needed)
        page.goto("https://www.monster.co.uk/")
        
        # get the page source
        page_source = page.content()
        print(page_source)
        
        # find the input field(s)
        
        
        # Fill in the search fields using preferences
        # page.fill('input[name="q"]', " OR ".join(preferences.job_titles))  # Job titles
        # page.fill('input[name="l"]', ", ".join(preferences.locations))     # Locations

        # # Submit the search form
        # page.click('button[type="submit"]')

        # Wait for results to load
        # page.wait_for_selector('.jobsearch-SerpJobCard')

        # # Scrape job listings matching preferences
        # job_cards = page.query_selector_all('.jobsearch-SerpJobCard')
        # jobs = []
        # for card in job_cards:
        #     title = card.query_selector('h2.title').inner_text()
        #     company = card.query_selector('.company').inner_text() if card.query_selector('.company') else ""
        #     location = card.query_selector('.location').inner_text() if card.query_selector('.location') else ""
        #     description = card.query_selector('.summary').inner_text() if card.query_selector('.summary') else ""
        #     # Add more fields as needed

        #     # You can add filtering logic here if needed!
        #     jobs.append({
        #         "title": title,
        #         "company": company,
        #         "location": location,
        #         "description": description
        #     })
        
        # take a screenshot
        page.screenshot(path="screenshot.png")

        page.pause()
        browser.close()
        return []

# Example usage:
# jobs = find_job_listings_with_playwright(preferences)
# print(jobs)