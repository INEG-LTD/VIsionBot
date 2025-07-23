# job_app_automation/main.py

from playwright.sync_api import sync_playwright
from time import sleep

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.linkedin.com/jobs/search/?currentJobId=3615666666&keywords=python&location=United%20States")
        print(page.title())
        sleep(10)
        browser.close()

if __name__ == "__main__":
    run()
