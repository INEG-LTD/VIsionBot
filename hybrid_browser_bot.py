#!/usr/bin/env python3
"""
Hybrid Browser Bot

A browser automation bot that uses Playwright for web interactions
and AppleScript for system-level tasks. This combines the best of both worlds:
- Playwright for reliable web element interaction
- AppleScript for system-level automation when needed
"""

import time
import subprocess
import os
import json as json_module
import random
import string
from typing import List, Dict, Optional, Any
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from pydantic import BaseModel
from page_detector import PageDetector, PageType
from typing import Any, Dict, List, TypedDict
import json as json_module
from application_filler import ApplicationFiller

class JobListing(BaseModel):
    title: str
    company: str
    href: str
    selector: str

class JobListingPayload(BaseModel):
    job_listings: List[JobListing]
    
class HybridBrowserBot:
    def __init__(self, headless: bool = False, preferences: Dict[str, Any] = None):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
        self.page_detector = None
        self.max_restarts = 3
        self.restart_count = 0
        self.preferences = preferences
        
    def start_browser(self):
        """Start the Playwright browser using your default Chrome profile"""
        try:
            # Ensure we're not in an asyncio context
            import asyncio
            try:
                asyncio.get_running_loop()
                print("⚠️ Warning: Running in asyncio context, this may cause issues")
            except RuntimeError:
                pass  # No asyncio loop running, which is good
            
            self.playwright = sync_playwright().start()
            
            # Try to connect to existing Chrome instance first
            try:
                print("🔗 Attempting to connect to existing Chrome instance...")
                self.browser = self.playwright.chromium.connect_over_cdp("http://localhost:9222")
                pages = self.browser.pages
                if pages:
                    self.page = pages[0]
                else:
                    self.page = self.browser.new_page()
                print("✅ Connected to existing Chrome instance")
                
                # Initialize page detector
                self.page_detector = PageDetector(self.page)
                return True
                
            except Exception as connect_error:
                print(f"⚠️ Could not connect to existing Chrome: {connect_error}")
                print("🚀 Launching new Chrome instance...")
            
            # If connection fails, launch a new instance with a unique profile
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            chrome_user_data_dir = os.path.expanduser(f"~/Library/Application Support/Google/Chrome/Automation_{unique_id}")
            
            # Create automation profile directory if it doesn't exist
            os.makedirs(chrome_user_data_dir, exist_ok=True)
            
            # Launch Chrome browser with unique automation profile
            self.browser = self.playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_user_data_dir,
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=VizDisplayCompositor'
                ],
                channel="chrome"  # Use your installed Chrome
            )
            
            # Get the first page or create a new one
            pages = self.browser.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = self.browser.new_page()
            
            # Initialize page detector
            self.page_detector = PageDetector(self.page)
            
            print("✅ Playwright browser started with separate automation profile")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Playwright browser: {e}")
            # Try alternative approach without persistent context
            try:
                print("🔄 Trying alternative browser launch method...")
                self.browser = self.playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled'
                    ]
                )
                self.context = self.browser.new_context()
                self.page = self.context.new_page()
                
                # Initialize page detector
                self.page_detector = PageDetector(self.page)
                
                print("✅ Playwright browser started with alternative method")
                return True
                
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                return False
    
    def stop_browser(self):
        """Stop the Playwright browser"""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("✅ Playwright browser stopped")
        except Exception as e:
            print(f"❌ Error stopping browser: {e}")
    
    def open_url(self, url: str) -> bool:
        """Open URL using Playwright"""
        try:
            print(f"📍 Navigating to: {url}")
            self.page.goto(url, wait_until='networkidle')
            self.current_url = url
            time.sleep(2)  # Wait for page to fully load
            print(f"✅ Successfully navigated to: {url}")
            return True
        except Exception as e:
            print(f"❌ Failed to navigate to {url}: {e}")
            return False
    
    def find_input_fields(self) -> List[Dict[str, Any]]:
        """Find all input fields using Playwright"""
        try:
            # Find all input elements
            inputs = self.page.query_selector_all('input, textarea, [contenteditable="true"]')
            
            input_fields = []
            for i, input_elem in enumerate(inputs):
                try:
                    # Get input properties
                    input_type = input_elem.get_attribute('type') or 'text'
                    placeholder = input_elem.get_attribute('placeholder') or ''
                    name = input_elem.get_attribute('name') or ''
                    id_attr = input_elem.get_attribute('id') or ''
                    value = input_elem.input_value() if input_elem.get_attribute('type') != 'file' else ''
                    
                    # Get visible text (label or aria-label)
                    label = ''
                    try:
                        # Try to find associated label
                        if id_attr:
                            label_elem = self.page.query_selector(f'label[for="{id_attr}"]')
                            if label_elem:
                                label = label_elem.inner_text().strip()
                        
                        # Try aria-label
                        if not label:
                            aria_label = input_elem.get_attribute('aria-label')
                            if aria_label:
                                label = aria_label
                        
                        # Try to find nearby text
                        if not label:
                            # Look for text within 100px of the input
                            bbox = input_elem.bounding_box()
                            if bbox:
                                nearby_text = self.page.query_selector_all('text')
                                for text_elem in nearby_text:
                                    text_bbox = text_elem.bounding_box()
                                    if text_bbox:
                                        distance = abs(bbox['y'] - text_bbox['y'])
                                        if distance < 50:  # Within 50px vertically
                                            text_content = text_elem.inner_text().strip()
                                            if text_content and len(text_content) < 100:
                                                label = text_content
                                                break
                    except:
                        pass
                    
                    input_fields.append({
                        'index': i,
                        'type': input_type,
                        'name': name,
                        'id': id_attr,
                        'placeholder': placeholder,
                        'value': value,
                        'label': label,
                        'element': input_elem
                    })
                    
                except Exception as e:
                    print(f"Error processing input {i}: {e}")
                    continue
            
            print(f"✅ Found {len(input_fields)} input fields using Playwright")
            return input_fields
            
        except Exception as e:
            print(f"❌ Error finding input fields: {e}")
            return []
    

    
    def find_submit_button(self) -> Optional[Dict[str, Any]]:
        """Find the most likely submit button using Playwright"""
        try:
            # Find all buttons and clickable elements
            buttons = self.page.query_selector_all('button, input[type="submit"], input[type="button"], [role="button"], a[href="#"], [onclick]')
            
            if not buttons:
                print("❌ No buttons found on the page")
                return None
            
            print(f"🔍 Analyzing {len(buttons)} buttons for submit button...")
            
            # Score each button based on submit likelihood
            submit_keywords = ['submit', 'search', 'find', 'go', 'apply', 'continue', 'next', 'save', 'send', 'post']
            
            scored_buttons = []
            for i, button in enumerate(buttons):
                try:
                    score = 0
                    
                    # Get button properties
                    button_type = button.get_attribute('type') or ''
                    button_text = button.inner_text().strip().lower()
                    button_value = button.get_attribute('value') or ''
                    button_name = button.get_attribute('name') or ''
                    button_id = button.get_attribute('id') or ''
                    button_class = button.get_attribute('class') or ''
                    button_role = button.get_attribute('role') or ''
                    
                    # Check if button is visible and enabled
                    is_visible = button.is_visible()
                    is_enabled = not button.get_attribute('disabled')
                    
                    if not is_visible or not is_enabled:
                        continue  # Skip invisible or disabled buttons
                    
                    # Combine all text for analysis
                    all_text = f"{button_text} {button_value} {button_name} {button_id} {button_class}".lower()
                    
                    # Check for submit-related keywords
                    for keyword in submit_keywords:
                        if keyword in all_text:
                            score += 10
                    
                    # Prefer submit type buttons
                    if button_type == 'submit':
                        score += 15
                    
                    # Prefer buttons with text content
                    if button_text:
                        score += 5
                    
                    # Prefer buttons with common submit text
                    submit_texts = ['submit', 'search', 'find jobs', 'apply', 'continue', 'go']
                    for submit_text in submit_texts:
                        if submit_text in button_text:
                            score += 8
                    
                    # Prefer buttons with action-oriented text
                    action_words = ['search', 'find', 'apply', 'submit', 'go', 'continue', 'next']
                    for word in action_words:
                        if word in button_text:
                            score += 5
                    
                    # Prefer buttons that are prominently positioned (first few buttons)
                    if i < 5:
                        score += 2
                    
                    # Prefer buttons with specific roles
                    if button_role == 'button':
                        score += 3
                    
                    # Prefer buttons with common submit IDs/classes
                    submit_identifiers = ['submit', 'search', 'apply', 'btn-submit', 'btn-search']
                    for identifier in submit_identifiers:
                        if identifier in button_id.lower() or identifier in button_class.lower():
                            score += 6
                    
                    # Special handling for job sites
                    if 'monster' in self.current_url.lower():
                        monster_keywords = ['search', 'find jobs', 'apply', 'submit']
                        for keyword in monster_keywords:
                            if keyword in button_text:
                                score += 10
                    
                    if 'indeed' in self.current_url.lower():
                        indeed_keywords = ['find jobs', 'search', 'apply']
                        for keyword in indeed_keywords:
                            if keyword in button_text:
                                score += 10
                    
                    if 'linkedin' in self.current_url.lower():
                        linkedin_keywords = ['search', 'apply', 'easy apply']
                        for keyword in linkedin_keywords:
                            if keyword in button_text:
                                score += 10
                    
                    # Create button info
                    button_info = {
                        'index': i,
                        'type': button_type,
                        'text': button_text,
                        'value': button_value,
                        'name': button_name,
                        'id': button_id,
                        'class': button_class,
                        'role': button_role,
                        'element': button,
                        'score': score
                    }
                    
                    scored_buttons.append(button_info)
                    
                except Exception as e:
                    print(f"Error processing button {i}: {e}")
                    continue
            
            if not scored_buttons:
                print("❌ No visible/enabled buttons found")
                return None
            
            # Sort by score (highest first)
            scored_buttons.sort(key=lambda x: x['score'], reverse=True)
            
            # Show top 3 candidates
            print("🏆 Top submit button candidates:")
            for i, button in enumerate(scored_buttons[:3]):
                print(f"  {i+1}. '{button['text']}' (score: {button['score']}) - type: {button['type']}")
            
            # Return the best button
            best_button = scored_buttons[0]
            print(f"✅ Best submit button: '{best_button['text']}' (score: {best_button['score']})")
            
            # Test if the button is clickable
            try:
                # Check if button is in viewport
                is_in_viewport = best_button['element'].is_visible()
                if not is_in_viewport:
                    print("🔄 Button not in viewport, scrolling into view...")
                    best_button['element'].scroll_into_view_if_needed()
                    time.sleep(0.5)
                
                # Verify button is still clickable
                if best_button['element'].is_enabled():
                    print("✅ Button is visible and enabled - ready to click")
                    return best_button
                else:
                    print("❌ Button is disabled")
                    return None
                    
            except Exception as e:
                print(f"❌ Error testing button clickability: {e}")
                return None
            
        except Exception as e:
            print(f"❌ Error finding submit button: {e}")
            return None
    
    def click_and_type_in_field(self, field: Dict[str, Any], text: str) -> bool:
        """Click on a field and type text using Playwright"""
        try:
            element = field['element']
            
            # Scroll element into view first
            element.scroll_into_view_if_needed()
            time.sleep(0.5)
            
            # Try to focus without clicking first (avoids overlay issues)
            try:
                element.focus()
                time.sleep(0.3)
            except:
                pass
            
            # Clear existing content
            element.fill('')
            time.sleep(0.2)
            
            # Type the text
            element.type(text, delay=50)  # 50ms delay between characters for realism
            time.sleep(0.5)
            
            print(f"✅ Successfully typed '{text}' in field")
            return True
            
        except Exception as e:
            print(f"❌ Error clicking and typing in field: {e}")
            # Try alternative approach - use keyboard navigation
            try:
                print("🔄 Trying alternative approach with keyboard navigation...")
                element.focus()
                time.sleep(0.3)
                element.fill(text)
                time.sleep(0.5)
                print(f"✅ Successfully typed '{text}' using alternative method")
                return True
            except Exception as e2:
                print(f"❌ Alternative approach also failed: {e2}")
                return False
    

    

    
    def find_job_listings(self) -> List[JobListing]:
        """Find job listings (title + company) that are clickable using GPT-4.1 first, then fallback to heuristics"""
        try:
            print("🔍 Finding job listings...")
            
            # Wait for job listings to load
            try:
                self.page.wait_for_load_state('networkidle', timeout=10000)  # 10 second timeout
                time.sleep(2)  # Extra wait for dynamic content
            except Exception as e:
                print(f"⚠️ Page load timeout, continuing anyway: {e}")
                time.sleep(2)
            
            # Try GPT-4.1 approach first
            print("🤖 Using GPT-4.1 to find job listings...")
            gpt_job_listings = self.find_job_listings_with_gpt()
            
            if gpt_job_listings:
                print(f"✅ GPT-4.1 found {len(gpt_job_listings)} job listings")
                # Limit to top 10 job listings to avoid too many API calls
                if len(gpt_job_listings) > 10:
                    print(f"⚠️ Limiting to top 10 job listings (found {len(gpt_job_listings)})")
                    gpt_job_listings = gpt_job_listings[:10]
                
                # Show found listings
                print("🏆 Job listings found:")
                for i, listing in enumerate(gpt_job_listings[:5]):
                    print(f"  {i+1}. '{listing.title}' at {listing.company}")
                
                return gpt_job_listings
            
            # Fallback to heuristics approach
            print("🔄 GPT-4.1 approach failed, falling back to heuristics...")
            return self.find_job_listings_with_heuristics()
            
        except Exception as e:
            print(f"❌ Error finding job listings: {e}")
            # Final fallback to heuristics
            return self.find_job_listings_with_heuristics()
    

    def find_job_listings_with_gpt(self) -> List[JobListing]:
        """Use GPT structured output to find job listings from the entire page source."""
        try:
            import openai
            client = openai.OpenAI()

            page_source = self.page.content()
            current_url = self.page.url

            user_prompt = f"""
    Analyze this HTML page source and extract all job listings. The page is from: {current_url}
    There are more than one job listing on the page.
    The button's selector element is very important. It is the element that is clicked to open the job details page.

    Return objects that have:
    - title: visible job title text
    - company: visible company text (empty string if truly absent)
    - href: absolute or relative link to details (use the anchor's href)
    - selector: a CSS selector that can re-find the element (use something stable like [data-*], ids, classes + :nth-of-type when needed)
    
    The HTML will have the job in a list, so you need to find the jobs and return them.

    HTML Source:
    {page_source}
    """

            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {
                        "role": "system",
                        "content": "You are an expert web scraper. Extract job listings from raw HTML and return them."
                    },
                    {"role": "user", "content": user_prompt}
                ],
                text_format=JobListingPayload,
            )

            # The Responses API returns a structured object; pull the text field.
            payload: JobListingPayload = response.output_parsed

            job_listings: List[JobListing] = []
            seen = set()

            for jl in payload.job_listings:
                try:
                    selector = jl.selector
                    if not selector:
                        continue

                    element = self.page.query_selector(selector)
                    if not element:
                        continue

                    # Ensure the clickable element
                    clickable = element
                    tag = element.evaluate('el => el.tagName.toLowerCase()')
                    if tag != 'a':
                        parent_link = element.evaluate('el => el.closest("a") && el.closest("a").getAttribute("href")')
                        if parent_link:
                            clickable = self.page.query_selector(f'a[href="{parent_link}"]')

                    title = jl.title.strip()
                    company = jl.company.strip()
                    href = jl.href.strip()

                    if not title or not href or not clickable:
                        continue

                    key = (title.lower(), company.lower())
                    if key in seen:
                        continue
                    seen.add(key)

                    job_listings.append(JobListing(title=title, company=company, href=href, selector=selector))
                except Exception as e:
                    print(f"⚠️ Error processing structured listing: {e}")
                    continue

            print(f"✅ Structured output found {len(job_listings)} job listings")
            return job_listings

        except Exception as e:
            print(f"❌ Error in GPT job listing detection: {e}")
            return []


    def find_job_listings_with_heuristics(self) -> List[Dict[str, Any]]:
        """Find job listings using the original heuristics approach"""
        try:
            print("🔍 Using heuristics to find job listings...")
            
            # Common selectors for job listings on different sites
            job_selectors = [
                # Indeed
                '[data-testid="jobTitle"]',
                '.job_seen_beacon',
                '.job_seen_beacon a',
                '[data-jk]',
                '.jobTitle',
                '.jobTitle a',
                '.job_seen_beacon',
                '.job_seen_beacon a',
                
                # Monster
                '.job-cardstyle__JobCardComponent',
                '.job-card a',
                '.job-title',
                '.job-title a',
                '.job-card',
                '.job-card a',
                
                # LinkedIn
                '.job-card-container',
                '.job-card-container a',
                '.job-card-list__title',
                '.job-card-list__title a',
                
                # Generic
                '[class*="job"] a',
                '[class*="position"] a',
                '[class*="listing"] a',
                '.job-listing a',
                '.position a',
                'a[href*="job"]',
                'a[href*="position"]',
                'a[href*="career"]',
                'a[href*="viewjob"]',
                'a[href*="job-details"]'
            ]
            
            job_listings = []
            
            for selector in job_selectors:
                try:
                    elements = self.page.query_selector_all(selector)
                    for element in elements:
                        try:
                            # Get the clickable element (either the element itself or its parent link)
                            clickable = element
                            if element.evaluate('el => el.tagName.toLowerCase()') != 'a':
                                # Find parent link
                                parent_link = element.evaluate('el => el.closest("a")')
                                if parent_link:
                                    clickable = self.page.query_selector(f'[href="{parent_link.getAttribute("href")}"]')
                            
                            if not clickable:
                                continue
                            
                            # Get job title and company info
                            job_title = ""
                            company_name = ""
                            
                            # Try to get job title from the element
                            job_title = element.inner_text().strip()
                            
                            # Try to find company name in nearby elements
                            try:
                                # Look for company in parent container
                                parent = element.evaluate('el => el.closest("[class*=\\"job\\"], [class*=\\"position\\"], [class*=\\"listing\\"]")')
                                if parent:
                                    company_selectors = [
                                        '[class*="company"]',
                                        '[class*="employer"]',
                                        '[class*="organization"]',
                                        '.company',
                                        '.employer',
                                        '.organization'
                                    ]
                                    for company_selector in company_selectors:
                                        company_elem = parent.query_selector(company_selector)
                                        if company_elem:
                                            company_name = company_elem.inner_text().strip()
                                            break
                            except:
                                pass
                            
                            # If no company found, try to extract from job title text
                            if not company_name and job_title:
                                # Look for patterns like "Job Title at Company" or "Job Title - Company"
                                if ' at ' in job_title:
                                    parts = job_title.split(' at ')
                                    if len(parts) >= 2:
                                        job_title = parts[0].strip()
                                        company_name = parts[1].strip()
                                elif ' - ' in job_title:
                                    parts = job_title.split(' - ')
                                    if len(parts) >= 2:
                                        job_title = parts[0].strip()
                                        company_name = parts[1].strip()
                            
                            # Skip if no meaningful job title
                            if not job_title or len(job_title) < 5:
                                continue
                            
                            # Get href
                            href = clickable.get_attribute('href') or ''
                            
                            # Vet the job listing with GPT-4.1
                            if self.vet_job_listing_with_gpt(job_title, company_name, href):
                                # Create job listing info
                                job_listing = {
                                    'title': job_title,
                                    'company': company_name,
                                    'href': href,
                                    'element': clickable,
                                    'full_text': f"{job_title} at {company_name}" if company_name else job_title
                                }
                                
                                # Avoid duplicates
                                if not any(existing['title'] == job_title and existing['company'] == company_name 
                                         for existing in job_listings):
                                    job_listings.append(job_listing)
                                
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    continue
            
            # If no job listings found with specific selectors, try a broader approach
            if not job_listings:
                print("🔍 Trying broader job listing detection...")
                
                # Look for all links that might be job listings
                all_links = self.page.query_selector_all('a')
                
                for link in all_links:
                    try:
                        text = link.inner_text().strip()
                        href = link.get_attribute('href') or ''
                        
                        # Skip navigation and non-job links
                        if not text or len(text) < 10 or len(text) > 200:
                            continue
                            
                        # Skip obvious navigation
                        nav_keywords = ['home', 'about', 'contact', 'login', 'sign up', 'help', 'support', 'privacy', 'terms']
                        if any(nav in text.lower() for nav in nav_keywords):
                            continue
                        
                        # Look for job-like patterns
                        job_patterns = [
                            'engineer', 'developer', 'programmer', 'analyst', 'manager', 'specialist',
                            'software', 'full-time', 'part-time', 'remote', 'senior', 'junior'
                        ]
                        
                        if any(pattern in text.lower() for pattern in job_patterns):
                            # Try to extract title and company
                            job_title = text
                            company_name = ""
                            
                            if ' at ' in text:
                                parts = text.split(' at ')
                                if len(parts) >= 2:
                                    job_title = parts[0].strip()
                                    company_name = parts[1].strip()
                            elif ' - ' in text:
                                parts = text.split(' - ')
                                if len(parts) >= 2:
                                    job_title = parts[0].strip()
                                    company_name = parts[1].strip()
                            
                            # Vet the job listing with GPT-4.1
                            if self.vet_job_listing_with_gpt(job_title, company_name, href):
                                job_listing = {
                                    'title': job_title,
                                    'company': company_name,
                                    'href': href,
                                    'element': link,
                                    'full_text': f"{job_title} at {company_name}" if company_name else job_title
                                }
                                
                                if not any(existing['title'] == job_title and existing['company'] == company_name 
                                         for existing in job_listings):
                                    job_listings.append(job_listing)
                                
                    except Exception as e:
                        continue
            
            print(f"✅ Heuristics found {len(job_listings)} job listings")
            if job_listings:
                print("🏆 Job listings found:")
                for i, listing in enumerate(job_listings[:5]):
                    print(f"  {i+1}. '{listing['title']}' at {listing['company']}")
            
            return job_listings
            
        except Exception as e:
            print(f"❌ Error in heuristics job listing detection: {e}")
            return []
            
        except Exception as e:
            print(f"❌ Error finding job listings: {e}")
            return []
    

    
    def evaluate_job_fit_with_gpt(self, job_description: str) -> Dict[str, Any]:
        """Use GPT-4.1 to evaluate if a job is a good fit based on preferences"""
        try:
            import openai
            
            # Create the evaluation prompt
            prompt = f"""
            Analyze this job description and evaluate if it's a good fit based on the candidate's preferences.
            
            JOB DESCRIPTION:
            {job_description[:1500]}  # Limit to first 1500 characters
            
            CANDIDATE PREFERENCES:
            - Job Titles: {self.preferences.get('job_titles', [])}
            - Locations: {self.preferences.get('locations', [])}
            - Salary Minimum: ${self.preferences.get('salary_min', 'Not specified')}
            - Employment Types: {self.preferences.get('employment_types', [])}
            - Required Skills: {self.preferences.get('required_skills', [])}
            - Experience Levels: {self.preferences.get('experience_levels', [])}
            - Visa Sponsorship Required: {self.preferences.get('visa_sponsorship_required', False)}
            - Remote Flexibility: {self.preferences.get('remote_flexibility', [])}
            - Desired Benefits: {self.preferences.get('desired_benefits', [])}
            - Exclude Keywords: {self.preferences.get('exclude_keywords', [])}
            
            Please provide a JSON response with:
            1. "overall_score": Score from 1-10 indicating overall fit
            2. "recommendation": "strong_yes", "yes", "maybe", "no", or "strong_no"
            3. "reasoning": Brief explanation of the recommendation
            4. "matched_preferences": List of preferences that match well
            5. "mismatched_preferences": List of preferences that don't match
            6. "salary_match": "above", "below", "unknown", or "not_specified"
            7. "location_match": "yes", "no", or "unknown"
            8. "skills_match": "strong", "moderate", "weak", or "unknown"
            
            Focus on:
            - Job title relevance
            - Location compatibility
            - Salary expectations
            - Required skills alignment
            - Employment type match
            - Remote work options
            - Visa sponsorship if needed
            
            Return only valid JSON.
            """
            
            # Get OpenAI client (you'll need to set up your API key)
            try:
                client = openai.OpenAI()
                
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are an expert job matching assistant. Analyze job descriptions and provide detailed matching scores based on candidate preferences."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                result = response.choices[0].message.content.strip()
                
                # Clean up markdown code blocks if present
                if result.startswith('```json'):
                    result = result[7:]  # Remove ```json
                if result.startswith('```'):
                    result = result[3:]   # Remove ```
                if result.endswith('```'):
                    result = result[:-3]  # Remove trailing ```
                
                result = result.strip()
                
                # Parse JSON response
                evaluation = json_module.loads(result)
                
                print(f"🤖 GPT-4.1 Evaluation: {evaluation.get('recommendation', 'unknown')} (score: {evaluation.get('overall_score', 'unknown')}/10)")
                print(f"   Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
                
                return evaluation
                
            except Exception as gpt_error:
                print(f"❌ GPT-4.1 evaluation failed: {gpt_error}")
                # Fallback to simple keyword matching
                return self.evaluate_job_fit_simple(job_description)
                
        except Exception as e:
            print(f"❌ Error in job fit evaluation: {e}")
            return self.evaluate_job_fit_simple(job_description)
    
    def evaluate_job_fit_simple(self, job_description: str) -> Dict[str, Any]:
        """Simple keyword-based job fit evaluation as fallback"""
        try:
            score = 0
            matched_preferences = []
            mismatched_preferences = []
            
            job_desc_lower = job_description.lower()
            
            # Check job titles
            if self.preferences.get('job_titles'):
                for title in self.preferences['job_titles']:
                    if title.lower() in job_desc_lower:
                        score += 20
                        matched_preferences.append('job_titles')
                        break
            
            # Check locations
            if self.preferences.get('locations'):
                for location in self.preferences['locations']:
                    if location.lower() in job_desc_lower:
                        score += 15
                        matched_preferences.append('locations')
                        break
            
            # Check required skills
            if self.preferences.get('required_skills'):
                skills_matched = 0
                for skill in self.preferences['required_skills']:
                    if skill.lower() in job_desc_lower:
                        skills_matched += 1
                
                if skills_matched > 0:
                    score += (skills_matched / len(self.preferences['required_skills'])) * 20
                    matched_preferences.append('required_skills')
            
            # Check for remote work
            if self.preferences.get('remote_flexibility'):
                remote_keywords = ['remote', 'work from home', 'telecommute', 'flexible']
                if any(keyword in job_desc_lower for keyword in remote_keywords):
                    score += 10
                    matched_preferences.append('remote_flexibility')
            
            # Check for excluded keywords
            if self.preferences.get('exclude_keywords'):
                for exclude in self.preferences['exclude_keywords']:
                    if exclude.lower() in job_desc_lower:
                        score -= 30
                        mismatched_preferences.append(f'excluded: {exclude}')
            
            # Determine recommendation
            if score >= 40:
                recommendation = "strong_yes"
            elif score >= 25:
                recommendation = "yes"
            elif score >= 10:
                recommendation = "maybe"
            elif score >= 0:
                recommendation = "no"
            else:
                recommendation = "strong_no"
            
            return {
                "overall_score": min(score, 10),
                "recommendation": recommendation,
                "reasoning": f"Simple keyword matching: {score} points",
                "matched_preferences": matched_preferences,
                "mismatched_preferences": mismatched_preferences,
                "salary_match": "unknown",
                "location_match": "yes" if "locations" in matched_preferences else "no",
                "skills_match": "strong" if "required_skills" in matched_preferences else "weak"
            }
            
        except Exception as e:
            print(f"❌ Error in simple job fit evaluation: {e}")
            return {
                "overall_score": 0,
                "recommendation": "unknown",
                "reasoning": "Evaluation failed",
                "matched_preferences": [],
                "mismatched_preferences": [],
                "salary_match": "unknown",
                "location_match": "unknown",
                "skills_match": "unknown"
            }
    
    def find_apply_button(self) -> Optional[Dict[str, Any]]:
        """Find the apply button on a job page using GPT-4.1"""
        try:
            print("🔍 Looking for apply button using GPT-4.1...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use GPT-4.1 to find the apply button
            apply_button_info = self._find_apply_button_with_gpt(page_html)
            
            if apply_button_info:
                # Try to find the actual element using the selector from GPT
                try:
                    element = self.page.query_selector(apply_button_info['selector'])
                    if element:
                        # Get the href if it's a link
                        href = None
                        try:
                            href = element.get_attribute('href')
                        except:
                            pass
                        
                        print(f"✅ GPT-4.1 found apply button: '{apply_button_info['text']}' (selector: {apply_button_info['selector']}, href: {href})")
                        return {
                            'text': apply_button_info['text'],
                            'element': element,
                            'selector': apply_button_info['selector'],
                            'href': href
                        }
                except Exception as e:
                    print(f"⚠️ Could not find element with selector '{apply_button_info['selector']}': {e}")
            
            print("❌ GPT-4.1 could not find apply button")
            return None
            
        except Exception as e:
            print(f"❌ Error finding apply button with GPT-4.1: {e}")
            return None
    
    def _find_apply_button_with_gpt(self, page_html: str) -> Optional[Dict[str, Any]]:
        """Use GPT-4.1 to find apply button from page HTML"""
        try:
            import openai
            import json
            
            # Create the prompt for GPT-4.1
            prompt = f"""
You are an expert at analyzing job posting pages and finding apply buttons. 

Given the HTML content of a job posting page, your task is to identify the main apply button that users would click to start the job application process.

Instructions:
1. Look for buttons or links that would initiate a job application
2. Common apply button text includes: "Apply", "Apply Now", "Easy Apply", "Quick Apply", "Submit Application", "Start Application"
3. Focus on buttons that are prominently displayed and clearly meant for applying
4. Return the information in JSON format with these fields:
   - text: The button text/label
   - selector: A CSS selector that would uniquely identify this button (prefer data-testid, id, or specific class combinations)
   - type: Either "button" or "link" depending on the element type

5. If you cannot find a clear apply button, return null
6. The button should be together with the focused job
7. There will be other apply buttons on the page, but we are looking for the one that is together with the focused job

Here is the HTML content of the page:

{page_html}

Find the apply button and return the result as JSON:
"""
            
            # Get OpenAI client
            client = openai.OpenAI()
            
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an apply button detection expert. Return only valid JSON with text, selector, and type fields."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Extract the result
            result = response.choices[0].message.content.strip()
            
            # Parse JSON result
            try:
                if result and result.lower() != "null":
                    # Clean up the result (remove markdown code blocks if present)
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    
                    apply_button_data = json.loads(result.strip())
                    
                    # Validate required fields
                    if 'text' in apply_button_data and 'selector' in apply_button_data:
                        return apply_button_data
                    else:
                        print("⚠️ GPT response missing required fields")
                        return None
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse GPT response as JSON: {e}")
                return None
                
        except Exception as e:
            print(f"❌ Error in GPT-4.1 apply button detection: {e}")
            return None
    
    def click_apply_button(self, apply_button: Dict[str, Any]) -> bool:
        """Click the apply button to start application process"""
        try:
            print(f"🎯 Clicking apply button: '{apply_button['text']}'")
            
            # Scroll into view and click
            apply_button['element'].scroll_into_view_if_needed()
            apply_button['element'].click()
            
            # Wait for application form to load
            time.sleep(3)
            
            print(f"✅ Successfully clicked apply button")
            return True
            
        except Exception as e:
            print(f"❌ Error clicking apply button: {e}")
            return False
    

    
    def extract_job_title_from_page(self) -> Optional[str]:
        """Extract job title from the current page using GPT-4.1"""
        try:
            print("📋 Extracting job title from current page using GPT-4.1...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use GPT-4.1 to extract the job title
            job_title = self._extract_job_title_with_gpt(page_html)
            
            if job_title and job_title != "Unknown Job Title":
                print(f"✅ GPT-4.1 extracted job title: '{job_title}'")
                return job_title
            else:
                print("❌ GPT-4.1 could not extract job title")
                return "Unknown Job Title"
                
        except Exception as e:
            print(f"❌ Error extracting job title with GPT-4.1: {e}")
            return "Unknown Job Title"
    
    def _extract_job_title_with_gpt(self, page_html: str) -> str:
        """Use GPT-4.1 to extract job title from page HTML"""
        try:
            import openai
            
            # Create the prompt for GPT-4.1
            prompt = f"""
You are an expert at analyzing job posting pages and extracting job titles. 

Given the HTML content of a job posting page, your task is to identify and extract the main job title that is currently in focus or being displayed.

Instructions:
1. Look for the most prominent job title on the page
2. Focus on headings (h1, h2, h3), title elements, or elements with job-related classes
3. The job title should be the specific position being advertised (e.g., "Senior Software Engineer", "Product Manager", "Data Analyst")
4. Avoid company names, location information, or other metadata
5. Return ONLY the job title as a clean string, nothing else
6. If you cannot find a clear job title, return "Unknown Job Title"

Here is the HTML content of the page:

{page_html}

Extract the job title:
"""
            
            # Get OpenAI client
            client = openai.OpenAI()
            
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a job title extraction expert. Return only the job title as a clean string."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Extract the result
            result = response.choices[0].message.content.strip()
            
            # Clean up the result
            if result and result != "Unknown Job Title":
                # Remove any quotes or extra formatting
                result = result.strip('"\'')
                # Limit length
                if len(result) > 200:
                    result = result[:200]
                return result
            else:
                return "Unknown Job Title"
                
        except Exception as e:
            print(f"❌ Error in GPT-4.1 job title extraction: {e}")
            return "Unknown Job Title"
    
    def extract_job_description_from_page(self) -> Optional[str]:
        """Extract job description from the current page using GPT-4.1"""
        try:
            print("📄 Extracting job description from current page using GPT-4.1...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use GPT-4.1 to extract the job description
            job_description = self._extract_job_description_with_gpt(page_html)
            
            if job_description:
                print(f"✅ GPT-4.1 extracted job description ({len(job_description)} characters)")
                return job_description
            else:
                print("❌ GPT-4.1 could not extract job description")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting job description with GPT-4.1: {e}")
            return None
    
    def _extract_job_description_with_gpt(self, page_html: str) -> Optional[str]:
        """Use GPT-4.1 to extract job description from page HTML"""
        try:
            import openai
            
            # Create the prompt for GPT-4.1
            prompt = f"""
You are an expert at analyzing job posting pages and extracting job descriptions. 

Given the HTML content of a job posting page, your task is to identify and extract the main job description that details the role, responsibilities, requirements, and qualifications.

Instructions:
1. Look for the main job description content on the page
2. Focus on sections that describe the role, responsibilities, requirements, qualifications, and benefits
3. Include key information such as:
   - Job responsibilities and duties
   - Required skills and qualifications
   - Preferred qualifications
   - Education requirements
   - Experience requirements
   - Benefits and perks (if mentioned)
4. Exclude navigation elements, headers, footers, and other non-job-related content
5. Return the job description as a clean, well-formatted text
6. If you cannot find a clear job description, return "No job description found"

Here is the HTML content of the page:

{page_html}

Extract the job description:
"""
            
            # Get OpenAI client
            client = openai.OpenAI()
            
            # Make the API call
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a job description extraction expert. Return only the job description as clean, well-formatted text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract the result
            result = response.choices[0].message.content.strip()
            
            # Clean up the result
            if result and result != "No job description found":
                # Remove any extra formatting but keep line breaks for readability
                result = result.strip()
                # Limit length to reasonable size (4000 characters)
                if len(result) > 4000:
                    result = result[:4000] + "..."
                return result
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error in GPT-4.1 job description extraction: {e}")
            return None
    

    
    def take_screenshot(self, filename: Optional[str] = None) -> str:
        """Take screenshot using Playwright"""
        if not filename:
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            filename = f"{random_string}.png"
        
        desktop_path = os.path.expanduser("~/Desktop")
        file_path = os.path.join(desktop_path, filename)
        
        try:
            self.page.screenshot(path=file_path, full_page=True)
            print(f"✅ Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            return ""
    

    

    
    def find_relevant_fields_for_job_preferences(self) -> List[Dict[str, Any]]:
        """Find relevant fields for job preferences using Playwright"""
        try:
            # Find all input fields
            input_fields = self.find_input_fields()
            
            if not input_fields:
                print("❌ No input fields found")
                return []
            
            print(f"🔍 Analyzing {len(input_fields)} fields for job preferences...")
            
            # Score each field based on relevance
            scored_fields = []
            for field in input_fields:
                score = 0
                field_text = f"{field['label']} {field['placeholder']} {field['name']} {field['id']}".lower()
                
                # Check for job title/keyword fields
                job_title_keywords = ['job', 'jobs', 'keyword', 'title', 'position', 'role', 'search']
                for keyword in job_title_keywords:
                    if keyword in field_text:
                        score += 15
                
                # Check for location-related keywords
                location_keywords = ['location', 'city', 'state', 'country', 'where', 'place', 'area']
                for keyword in location_keywords:
                    if keyword in field_text:
                        score += 12
                
                # Check for salary-related keywords
                salary_keywords = ['salary', 'pay', 'compensation', 'wage', 'hourly', 'annual', 'rate']
                for keyword in salary_keywords:
                    if keyword in field_text:
                        score += 10
                
                # Check for employment type-related keywords
                employment_keywords = ['full-time', 'part-time', 'contract', 'temporary', 'internship', 'freelance', 'remote']
                for keyword in employment_keywords:
                    if keyword in field_text:
                        score += 8
                
                # Check for experience-related keywords
                experience_keywords = ['experience', 'level', 'seniority', 'years', 'entry', 'mid', 'senior']
                for keyword in experience_keywords:
                    if keyword in field_text:
                        score += 7
                
                # Check for skills-related keywords
                skills_keywords = ['skills', 'technologies', 'languages', 'tools', 'frameworks']
                for keyword in skills_keywords:
                    if keyword in field_text:
                        score += 6
                
                # Check for company-related keywords
                company_keywords = ['company', 'employer', 'organization', 'firm']
                for keyword in company_keywords:
                    if keyword in field_text:
                        score += 5
                
                # Check for industry-related keywords
                industry_keywords = ['industry', 'sector', 'field', 'domain']
                for keyword in industry_keywords:
                    if keyword in field_text:
                        score += 5
                
                # Check for remote/work flexibility keywords
                remote_keywords = ['remote', 'flexibility', 'telecommuting', 'work from home', 'hybrid']
                for keyword in remote_keywords:
                    if keyword in field_text:
                        score += 8
                
                # Check for visa/sponsorship keywords
                visa_keywords = ['visa', 'sponsorship', 'work permit', 'immigration']
                for keyword in visa_keywords:
                    if keyword in field_text:
                        score += 6
                
                # Prefer text inputs over other types
                if field['type'] in ['text', 'search', '']:
                    score += 3
                
                # Prefer empty fields (ready to be filled)
                if not field['value']:
                    score += 2
                
                # Prefer fields with placeholders (more descriptive)
                if field['placeholder']:
                    score += 1
                
                # Special handling for specific job sites
                if 'indeed' in self.current_url.lower():
                    indeed_keywords = ['what', 'where', 'job title', 'keywords']
                    for keyword in indeed_keywords:
                        if keyword in field_text:
                            score += 5
                
                if 'monster' in self.current_url.lower():
                    monster_keywords = ['keywords', 'job title', 'location']
                    for keyword in monster_keywords:
                        if keyword in field_text:
                            score += 5
                
                if 'linkedin' in self.current_url.lower():
                    linkedin_keywords = ['keywords', 'title', 'location']
                    for keyword in linkedin_keywords:
                        if keyword in field_text:
                            score += 5
                
                # Create field info with score
                field_info = {
                    **field,
                    'score': score,
                    'matched_preferences': []
                }
                
                # Track which preferences this field matches
                if self.preferences.get('job_titles') and score > 10:
                    field_info['matched_preferences'].append('job_titles')
                if self.preferences.get('locations') and 'location' in field_text:
                    field_info['matched_preferences'].append('locations')
                if self.preferences.get('salary_min') and 'salary' in field_text:
                    field_info['matched_preferences'].append('salary')
                if self.preferences.get('employment_types') and any(emp in field_text for emp in ['full-time', 'part-time', 'contract']):
                    field_info['matched_preferences'].append('employment_types')
                if self.preferences.get('remote_flexibility') and 'remote' in field_text:
                    field_info['matched_preferences'].append('remote_flexibility')
                
                scored_fields.append(field_info)
            
            # Sort by score (highest first)
            scored_fields.sort(key=lambda x: x['score'], reverse=True)
            
            # Show top 5 relevant fields
            print("🏆 Top relevant fields for job preferences:")
            for i, field in enumerate(scored_fields[:5]):
                field_name = field['label'] or field['placeholder'] or field['name'] or f"Field {field['index']}"
                print(f"  {i+1}. '{field_name}' (score: {field['score']}) - matches: {field['matched_preferences']}")
            
            # Return fields with scores above threshold
            relevant_fields = [field for field in scored_fields if field['score'] >= 5]
            
            print(f"✅ Found {len(relevant_fields)} relevant fields for job preferences")
            return relevant_fields
            
        except Exception as e:
            print(f"❌ Error finding relevant fields: {e}")
            return []
                    
    def run_job_application(self, job_site_url: str) -> bool:
        """
        Main automation function that orchestrates the entire job application process.
        
        This function implements a simplified state machine that:
        1. Detects the current page type (search, results, job detail, application, etc.)
        2. Handles the page directly using specialized page handler functions
        3. Each page handler performs the necessary actions and returns control
        4. Navigation between pages is handled automatically by form submissions and clicks
        
        The automation follows this general flow:
        - Search Page → Fill search form → Results Page
        - Results Page → Find job listings → Job Detail Page  
        - Job Detail Page → Evaluate job → Apply or continue to next job
        - Application Page → Fill application form → Continue
        
        Simplified Architecture:
        - Removed redundant action-based routing system
        - Page handlers now directly perform their actions
        - Main loop only handles page detection and stopping conditions
        """
        print(f"🚀 Starting job application for: {job_site_url}")
        
        try:
            # Initialize browser and navigate to the job site
            if not self.start_browser():
                return False
            if not self.open_url(job_site_url):
                return False
            
            # State tracking variables for the automation loop
            current_url = self.page.url                    # Track URL changes to detect page navigation
            current_page_type = None                       # Current page type (search, results, etc.)
            current_action = None                          # Current action being performed
            
            # Main automation loop - runs until completion or error
            max_iterations = 50  # Safety limit to prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 Iteration {iteration}/{max_iterations}")
                
                # PAGE NAVIGATION DETECTION
                # Check if the user/bot has navigated to a different page
                new_url = self.page.url
                if new_url != current_url:
                    # URL changed - we're on a new page, need to detect and handle it
                    print(f"📍 New page detected: {new_url}")
                    current_url = new_url
                    
                    # Detect the page type and handle it directly
                    # This uses the page detector to analyze the current page
                    page_result = self.detect_and_handle_page()
                    current_page_type = page_result.get('page_type', 'unknown')
                    
                    print(f"🎯 Page type: {current_page_type}")
                    
                    # Check if we should stop
                    if page_result.get('action') == 'stop':
                        print("🛑 Stopping automation")
                        return False
                        
                else:
                    # Same URL - check if this is the first time on this page
                    if current_page_type is None:
                        # First time on this page, need to detect what type it is
                        page_result = self.detect_and_handle_page()
                        current_page_type = page_result.get('page_type', 'unknown')
                        
                        print(f"🎯 Page type: {current_page_type}")
                        
                        # Check if we should stop
                        if page_result.get('action') == 'stop':
                            print("🛑 Stopping automation")
                            return False
                    else:
                        # Same page, no need to re-detect
                        print(f"🔄 Continuing on same page: {current_page_type}")
            
            print("⚠️ Max iterations reached")
            return False
            
        except Exception as e:
            print(f"❌ Error in job application: {e}")
            return False
        finally:
            if hasattr(self, 'page') and self.page:
                self.page.pause()
            else:
                print("⚠️ Browser not started, skipping pause")
    
    def _fill_search_form(self) -> bool:
        """Fill the job search form with preferences"""
        try:
            print("🔍 Finding relevant fields for search form...")
            relevant_fields = self.find_relevant_fields_for_job_preferences()
            
            if not relevant_fields:
                print("❌ No relevant fields found")
                return False
            
            print(f"🔍 Found {len(relevant_fields)} relevant fields")
            fields_filled = 0
            
            # Fill job title field
            if self.preferences.get('job_titles'):
                job_title_field = None
                for field in relevant_fields:
                    if ('job_titles' in field['matched_preferences'] or 
                        'job' in field.get('label', '').lower() or 
                        'keyword' in field.get('placeholder', '').lower()):
                        job_title_field = field
                        break
                
                if job_title_field:
                    search_term = self.preferences['job_titles'][0]
                    if self.click_and_type_in_field(job_title_field, search_term):
                        print(f"✅ Filled job title field: '{search_term}'")
                        fields_filled += 1
                    else:
                        print("❌ Failed to fill job title field")
            
            # Fill location field
            if self.preferences.get('locations'):
                location_field = None
                for field in relevant_fields:
                    if ('locations' in field['matched_preferences'] or 
                        'location' in field.get('label', '').lower() or 
                        'where' in field.get('placeholder', '').lower()):
                        location_field = field
                        break
                
                if location_field:
                    location = self.preferences['locations'][0]
                    if self.click_and_type_in_field(location_field, location):
                        print(f"✅ Filled location field: '{location}'")
                        fields_filled += 1
                    else:
                        print("❌ Failed to fill location field")
            
            # Fill salary field
            if self.preferences.get('salary_min'):
                salary_field = None
                for field in relevant_fields:
                    if ('salary' in field['matched_preferences'] or 
                        'salary' in field.get('label', '').lower()):
                        salary_field = field
                        break
                
                if salary_field:
                    salary = str(self.preferences['salary_min'])
                    if self.click_and_type_in_field(salary_field, salary):
                        print(f"✅ Filled salary field: '{salary}'")
                        fields_filled += 1
                    else:
                        print("❌ Failed to fill salary field")
            
            print(f"✅ Filled {fields_filled} fields with job preferences")
            
            # Submit the form
            submit_button_info = self.find_submit_button()
            if submit_button_info:
                try:
                    submit_button_info['element'].click()
                    time.sleep(3)
                    print(f"✅ Successfully clicked submit button: '{submit_button_info['text']}'")
                    self.take_screenshot("after_submit.png")
                    return True
                except Exception as e:
                    print(f"❌ Error clicking submit button: {e}")
                    return False
            else:
                print("❌ Could not find submit button")
                return False
                
        except Exception as e:
            print(f"❌ Error filling search form: {e}")
            return False
    
    def _process_job_listings(self, job_listings: List[JobListing]) -> bool:
        """Process a list of job listings"""
        try:
            for i, job_listing in enumerate(job_listings):
                print(f"\n📋 Processing job {i+1}/{len(job_listings)}: '{job_listing.title}' at {job_listing.company}")
                
                # Click on the job listing
                try:
                    job_css_selector = job_listing.selector
                    job_element = self.page.query_selector(job_css_selector)
                    job_element.scroll_into_view_if_needed()
                    job_element.click()
                    time.sleep(3)  # Wait for job details to load
                except Exception as e:
                    print(f"❌ Error clicking job listing: {e}")
                    continue
                
                # Check if we're now on a job detail page (either full page or panel)
                print("🔍 Checking if job details loaded...")
                
                print("✅ Proceeding with evaluation")
                # The job detail page handler will evaluate and open applications in new tabs
                self._handle_job_detail_page()
                
        except Exception as e:
            print(f"❌ Error processing job listings: {e}")
            return False
    

    
    def _fill_application_form(self) -> bool:
        """Fill out a job application form"""
        try:
            print("📝 Filling application form...")
            # This is a placeholder - will be implemented based on specific form structures
            # For now, just take a screenshot and pause for user
            self.take_screenshot("application_form.png")
            print("⏸️ Application form detected. Please fill it out manually and resume.")
            self.page.pause()
            return True
            
        except Exception as e:
            print(f"❌ Error filling application form: {e}")
            return False

    def detect_and_handle_page(self) -> Dict[str, Any]:
        """Detect current page type and handle it appropriately"""
        if not self.page_detector:
            print("❌ Page detector not initialized")
            return {'action': 'error', 'reason': 'page_detector_not_initialized'}
        
        # Detect page type
        detection_result = self.page_detector.detect_page_type()
        page_type = detection_result['page_type']
        
        # Add preferences to detection result if provided
        if self.preferences:
            detection_result['preferences'] = self.preferences
        
        # Handle different page types
        if page_type == PageType.SEARCH_PAGE:
            return self._handle_search_page(detection_result)
        elif page_type == PageType.RESULTS_PAGE:
            return self._handle_results_page(detection_result)
        elif page_type == PageType.JOB_DETAIL_PAGE:
            return self._handle_job_detail_page()
        elif page_type == PageType.APPLICATION_PAGE:
            return self._handle_application_page(detection_result)
        elif page_type == PageType.LOGIN_PAGE:
            return self._handle_login_page(detection_result)
        elif page_type == PageType.CAPTCHA_PAGE:
            return self._handle_captcha_page(detection_result)
        elif page_type == PageType.ERROR_PAGE:
            return self._handle_error_page(detection_result)
        else:
            return self._handle_unknown_page(detection_result)
    

    
    def _handle_search_page(self) -> Dict[str, Any]:
        """Handle job search form page"""
        print("🔍 Handling search page...")
        
        # Wait for the user to complete any login process needed
        print("⏸️ Waiting for user to complete login process if needed...")
        self.page.pause()
        
        # Resume when user is ready
        print("📝 Filling search form...")
        success = self._fill_search_form()
        if not success:
            print("❌ Failed to fill search form")
            return {'action': 'stop', 'page_type': 'search_page', 'reason': 'fill_form_failed'}
        
        # After filling form, we expect navigation to results page
        # The form submission should trigger navigation, so we return None to continue
        return {'action': None, 'page_type': 'search_page', 'reason': 'form_filled'}
    
    def _handle_results_page(self) -> Dict[str, Any]:
        """Handle job search results page"""
        print("📋 Handling results page...")
        
        job_listings = self.find_job_listings()
        if job_listings:
            print(f"✅ Found {len(job_listings)} job listings")
            # Process jobs
            success = self._process_job_listings(job_listings)
            if not success:
                print("❌ Failed to process job listings")
                return {'action': 'stop', 'page_type': 'results_page', 'reason': 'process_listings_failed'}
        else:
            print("⚠️ No job listings found")
            return {'action': 'stop', 'page_type': 'results_page', 'reason': 'no_listings_found'}
        
        # After processing jobs, we expect navigation to job detail page
        # The job processing should trigger navigation, so we return None to continue
        return {'action': None, 'page_type': 'results_page', 'reason': 'listings_processed'}
    
    def _handle_job_detail_page(self):
        """Handle individual job posting page"""
        print("📄 Handling job detail page...")
        
        try:
            # Extract job description from the page
            print("🔍 Extracting job description...")
            job_title = self.extract_job_title_from_page()
            job_description = self.extract_job_description_from_page()
            
            if not job_description:
                print("❌ Could not extract job description")
                return {'action': 'stop', 'page_type': 'job_detail_page', 'reason': 'no_job_description'}
            
            print(f"✅ Extracted job description ({len(job_description)} characters)")
            
            # Get current job info from page
            current_url = self.page.url
            
            print(f"📋 Job: {job_title}")
            print(f"🔗 URL: {current_url}")
            
            # Evaluate job fit
            print("🤖 Evaluating job fit...")
            evaluation = self.evaluate_job_fit_with_gpt(job_description)
            
            recommendation = evaluation.get('recommendation', 'unknown')
            score = evaluation.get('overall_score', 0)
            
            print(f"📊 Job evaluation: {recommendation} (score: {score}/10)")
            print(f"💭 Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
            
            # If it's a good match, open apply link in new tab
            if recommendation in ['strong_yes', 'yes']:
                print("🎯 Good match! Looking for apply button...")
                
                # Find apply button
                apply_button = self.find_apply_button()
                if apply_button:
                    print("✅ Found apply button, opening in new tab...")
                    
                    # Get the apply URL
                    apply_url = apply_button.get('href', '')
                    if not apply_url:
                        # Try to get href from the element
                        try:
                            apply_url = apply_button['element'].get_attribute('href')
                        except:
                            pass
                    
                    if apply_url:
                        # Open apply URL in new tab
                        print(f"🔗 Opening application in new tab: {apply_url}")
                        new_page = self.page.context.new_page()
                        new_page.goto(apply_url, wait_until='networkidle')
                        
                        # Take screenshot of application form
                        screenshot_path = new_page.screenshot(path=f"application_{job_title.replace(' ', '_')[:30]}.png")
                        print(f"📸 Screenshot saved: {screenshot_path}")
                        print("✅ Application opened in new tab - continuing to next job")
                        
                        # Check if we're in a panel view (job listings still visible)
                        job_listings = self.page.query_selector_all('[class*="job"], [class*="position"], [class*="listing"]')
                        if len(job_listings) > 0:
                            print("🔄 Job listings still visible - continuing to next job")
                            return {'action': None, 'page_type': 'job_detail_page', 'reason': 'good_match_continue'}
                        else:
                            print("🔄 Full job detail page - going back to results")
                            self.page.go_back()
                            time.sleep(2)
                            return {'action': None, 'page_type': 'results_page', 'reason': 'good_match_back'}
                    else:
                        # No URL available, click the button instead
                        print("🔗 No direct URL, clicking apply button...")
                        if self.click_apply_button(apply_button):
                            print("✅ Successfully clicked apply button!")
                            
                            # Wait for application form to load
                            time.sleep(3)
                            
                            # Take screenshot of application form
                            screenshot_path = self.take_screenshot(f"application_{job_title.replace(' ', '_')[:30]}.png")
                            print(f"📸 Screenshot saved: {screenshot_path}")
                            
                            print("✅ Application form opened - continuing to next job")
                        else:
                            print("❌ Failed to click apply button")
                else:
                    print("⚠️ No apply button found")
            else:
                print("❌ Job doesn't match preferences well enough")
                
        except Exception as e:
            print(f"❌ Error handling job detail page: {e}")
            return {'action': 'stop', 'page_type': 'job_detail_page', 'reason': str(e)}
        
        # Default return for job detail page - continue to next job
        return {'action': None, 'page_type': 'job_detail_page', 'reason': 'job_evaluated'}
    
    def _handle_application_page(self) -> Dict[str, Any]:
        """Handle job application form page"""
        print("📝 Handling application page...")
        
        print("📝 Filling application form...")
        success = self._fill_application_form()
        if not success:
            print("❌ Failed to fill application form")
            return {'action': 'stop', 'page_type': 'application_page', 'reason': 'fill_application_failed'}
        
        # After filling form, we expect navigation to next step
        return {'action': None, 'page_type': 'application_page', 'reason': 'application_filled'}
    
    def _handle_login_page(self) -> Dict[str, Any]:
        """Handle login/authentication page"""
        print("🔒 Handling login page...")
        print("⏸️ Pausing for user login. Please log in manually and resume.")
        self.page.pause()
        
        # After resume, check if login was successful
        time.sleep(2)
        new_detection = self.page_detector.detect_page_type()
        if new_detection['page_type'] == PageType.LOGIN_PAGE:
            print("❌ Still on login page. Please complete login and resume again.")
            self.page.pause()
        
        print("✅ Login appears complete. Continuing...")
        return {'action': None, 'page_type': 'login_page', 'reason': 'login_completed'}
    
    def _handle_captcha_page(self) -> Dict[str, Any]:
        """Handle CAPTCHA verification page"""
        print("🤖 Handling CAPTCHA page...")
        print("⏸️ CAPTCHA detected. Please solve the CAPTCHA manually and resume.")
        self.page.pause()
        
        # After resume, check if CAPTCHA was solved
        time.sleep(2)
        new_detection = self.page_detector.detect_page_type()
        if new_detection['page_type'] == PageType.CAPTCHA_PAGE:
            print("❌ Still on CAPTCHA page. Please solve the CAPTCHA and resume again.")
            self.page.pause()
        
        print("✅ CAPTCHA appears solved. Continuing...")
        return {'action': None, 'page_type': 'captcha_page', 'reason': 'captcha_solved'}
    
    def _handle_error_page(self) -> Dict[str, Any]:
        """Handle error pages"""
        print("❌ Handling error page...")
        return {'action': 'stop', 'page_type': 'error_page', 'reason': 'error_page_detected'}
    
    def _handle_unknown_page(self) -> Dict[str, Any]:
        """Handle unknown page types"""
        print("❓ Handling unknown page...")
        
        # Check if we've been on unknown pages too long
        if self.page_detector.should_restart():
            print("🔄 Too many unknown pages - considering restart")
            if self.restart_count < self.max_restarts:
                self.restart_count += 1
                # For now, just wait for user intervention instead of trying to restart
                print("⏸️ Too many unknown pages - waiting for user intervention...")
                self.page.pause()
                return {'action': None, 'page_type': 'unknown_page', 'reason': 'user_intervention_after_restart'}
            else:
                print("❌ Max restarts reached, stopping")
                return {'action': 'stop', 'page_type': 'restart', 'reason': 'max_restarts_reached'}
        
        # Wait for user intervention
        print("⏸️ Waiting for user intervention...")
        self.page.pause()
        return {'action': None, 'page_type': 'unknown_page', 'reason': 'user_intervention_completed'}


    def vet_job_listing_with_gpt(self, job_title: str, company_name: str = "", href: str = "") -> bool:
        """Use GPT-4.1 to determine if this is actually a job listing"""
        try:
            import openai
            
            # Quick pre-filter to avoid unnecessary API calls
            if self.vet_job_listing_simple(job_title, company_name, href) == False:
                return False
            
            # Create the vetting prompt
            prompt = f"""
            Determine if this is an actual job listing or just a navigation element/button.
            
            Job Title: "{job_title}"
            Company: "{company_name}"
            URL: "{href}"
            
            Consider:
            1. Does this look like a real job title (e.g., "Software Engineer", "Marketing Manager")?
            2. Is this a navigation element (e.g., "Sign Up", "Login", "About Us")?
            3. Is this a category or filter (e.g., "Full-time", "Remote", "Part-time")?
            4. Is this an action button (e.g., "Apply", "Save", "Share")?
            5. Does the URL contain job-related paths?
            
            Return only "YES" if this is an actual job listing, or "NO" if it's not.
            """
            
            try:
                client = openai.OpenAI()
                
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a job listing classifier. Determine if text represents an actual job posting or just navigation/UI elements."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10,
                    timeout=10  # 10 second timeout
                )
                
                result = response.choices[0].message.content.strip().upper()
                
                is_job = result == "YES"
                print(f"🤖 GPT-4.1 vetting: '{job_title}' -> {result} ({'Job' if is_job else 'Not a job'})")
                
                return is_job
                
            except Exception as gpt_error:
                print(f"❌ GPT-4.1 vetting failed: {gpt_error}")
                # Fallback to simple keyword-based vetting
                return self.vet_job_listing_simple(job_title, company_name, href)
                
        except Exception as e:
            print(f"❌ Error in job listing vetting: {e}")
            return self.vet_job_listing_simple(job_title, company_name, href)
    
    def vet_job_listing_simple(self, job_title: str, company_name: str = "", href: str = "") -> bool:
        """Simple keyword-based vetting as fallback"""
        try:
            # Convert to lowercase for comparison
            title_lower = job_title.lower()
            company_lower = company_name.lower()
            href_lower = href.lower()
            
            # Navigation/UI elements to exclude
            exclude_keywords = [
                'sign up', 'login', 'register', 'about', 'contact', 'help', 'support',
                'privacy', 'terms', 'cookies', 'settings', 'profile', 'account',
                'home', 'search', 'filter', 'sort', 'save', 'share', 'apply',
                'full-time', 'part-time', 'remote', 'hybrid', 'contract', 'temporary',
                'entry level', 'senior', 'junior', 'mid-level', 'freelance',
                'popular', 'trending', 'featured', 'new', 'hot', 'urgent',
                'employer', 'recruiter', 'post job', 'hire', 'advertise',
                'job tracker', 'saved jobs', 'recent searches', 'recommendations'
            ]
            
            # Check if title contains excluded keywords
            for keyword in exclude_keywords:
                if keyword in title_lower:
                    print(f"❌ Excluded '{job_title}' (contains '{keyword}')")
                    return False
            
            # Job title patterns to include
            include_patterns = [
                'engineer', 'developer', 'programmer', 'analyst', 'manager',
                'specialist', 'coordinator', 'assistant', 'director', 'lead',
                'architect', 'consultant', 'advisor', 'supervisor', 'officer',
                'representative', 'associate', 'executive', 'administrator'
            ]
            
            # Check if title contains job patterns
            has_job_pattern = any(pattern in title_lower for pattern in include_patterns)
            
            # Check if URL contains job-related paths
            job_url_patterns = ['/job/', '/position/', '/career/', '/viewjob', '/job-details']
            has_job_url = any(pattern in href_lower for pattern in job_url_patterns)
            
            # Must have either job pattern in title or job URL
            is_job = has_job_pattern or has_job_url
            
            print(f"🔍 Simple vetting: '{job_title}' -> {'Job' if is_job else 'Not a job'} (pattern: {has_job_pattern}, URL: {has_job_url})")
            
            return is_job
            
        except Exception as e:
            print(f"❌ Error in simple job vetting: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Ensure we're not in an asyncio context
    import asyncio
    try:
        asyncio.get_running_loop()
        print("❌ Error: This script cannot run inside an asyncio event loop")
        print("Please run this script directly, not from within an async context")
        exit(1)
    except RuntimeError:
        pass  # No asyncio loop running, which is good
    
    # Test preferences
    preferences = {
        'job_titles': ['ios engineer'],
        'locations': ['london'],
        'salary_min': 12000,
        'employment_types': ['Full-time'],
        'industries': None,  # None means any
        'company_sizes': None,  # None means any 
        'required_skills': None,  # None means any
        'experience_levels': ['Mid', 'Senior'],
        'visa_sponsorship_required': False,
        'remote_flexibility': ['Remote'],
        'desired_benefits': ['Health Insurance', 'Stock Options'],
        'exclude_keywords': ['unpaid', 'internship']
    }
    
    bot = HybridBrowserBot(headless=False, preferences=preferences)
    
    # Test the bot
    print("🧪 Testing Hybrid Browser Bot")
    print("=" * 50)
    
    success = bot.run_job_application("https://www.monster.co.uk")
    
    if success:
        print("✅ Job application test completed successfully!")
    else:
        print("❌ Job application test failed") 