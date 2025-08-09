#!/usr/bin/env python3
"""
Hybrid Browser Bot

A browser automation bot that uses Playwright for web interactions
and AppleScript for system-level tasks. This combines the best of both worlds:
- Playwright for reliable web element interaction
- AppleScript for system-level automation when needed
"""

from enum import Enum
import time
import subprocess
import os
import json as json_module
import random
import string
from typing import List, Dict, Optional, Any
from google import genai
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
from pydantic import BaseModel
from page_detector import PageDetector, PageType
from typing import Any, Dict, List, TypedDict
import json as json_module
from fill_and_submit_job_form import ApplicationFiller, FieldType, TextApplicationField, SubmitButtonApplicationField, TextApplicationFieldResponse
from job_detail_to_form_transition_handler import JobDetailToFormTransitionHandler

class JobListing(BaseModel):
    title: str
    company: str
    href: str
    selector: str
    
class JobListingDetails(BaseModel):
    job_listing: JobListing
    job_description: str
    job_fit: Dict[str, Any]
    job_url: str

class JobListingPayload(BaseModel):
    job_listings: List[JobListing]
    
class PageTypeResult(Enum):
    SUCCESS = "success"
    CONTINUE = "continue"
    ERROR = "error"
    
class FindJobsBot:
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
        self.accepted_job_listings: List[JobListingDetails] = []
        self.rejected_job_listings: List[JobListingDetails] = []
        
        self.max_jobs_to_find = 5
        
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
    
    def find_search_page_input_fields(self) -> TextApplicationFieldResponse:
        """Find all input fields using Playwright"""
        try:
            from google import genai
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the text input fields in the search page.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            Ensure that the input is not used for a dropdown or a radio button or a checkbox.
            The input should be contextually used to enter a value.
            Some text inputs are used for a dropdown or a radio button or a checkbox, do not include them.
            
            For a text input the possible values are a string
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job search.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            7. VERY IMPORTANT - Only include INTERACTIVE and VISIBLE fields:
                - Do NOT include fields that are hidden by CSS (display: none, visibility: hidden, opacity: 0)
                - Do NOT include fields that are positioned off-screen or have zero dimensions
                - Do NOT include fields that are disabled or not enabled
                - Do NOT include fields from future form steps that aren't visible yet
                - Only include fields that a user can actually see and interact with on the current page
            8. For each field, set field_is_visible=true and field_in_form=true only if the field is actually visible and interactive
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=TextApplicationFieldResponse
                )
            )
            
            all_fields = response.parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            # Parse the JSON response
            text_input_fields: List[TextApplicationField] = [field for field in all_fields if field.field_type == FieldType.TEXT]
            
            return TextApplicationFieldResponse(fields=text_input_fields)
        except Exception as e:
            print(f"❌ Error finding input fields: {e}")
            return TextApplicationFieldResponse(fields=[])
     
    def fill_text_input_fields(self, text_input_fields: TextApplicationFieldResponse) -> bool:
        """Fill the text input fields with the preferences"""
        try:
            for field in text_input_fields.fields:
                element = self.page.locator(field.field_selector)
                if element:
                    element.type(field.field_value)
                else:
                    print(f"❌ Element not found: {field.field_selector}")
            return True
        except Exception as e:
            print(f"❌ Error filling text input fields: {e}")
            return False
      
    def find_search_page_submit_button(self) -> Optional[SubmitButtonApplicationField]:
        """Find the most likely submit button using Playwright"""
        try:
            from google import genai
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            page_content = self.page.content()
            
            prompt = f"""
            Analyze this HTML and identify the most likely submit button in the search page.
            Return a JSON object with the SubmitButtonApplicationField object.
            
            CRITICAL GUIDELINES:
            1. Find the most likely submit button in the search page.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            7. VERY IMPORTANT - Only include INTERACTIVE and VISIBLE fields:
                - Do NOT include fields that are hidden by CSS (display: none, visibility: hidden, opacity: 0)
                - Do NOT include fields that are positioned off-screen or have zero dimensions
                - Do NOT include fields that are disabled or not enabled
            HTML of the page: {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=SubmitButtonApplicationField
                )
            )
            
            submit_button = response.parsed
            
            return submit_button
            
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
        """Find job listings (title + company) that are clickable using Gemini 2.5 Pro first, then fallback to heuristics"""
        try:
            print("🔍 Finding job listings...")
            
            print("🤖 Using Gemini 2.5 Pro to find job listings...")
            gpt_job_listings = self.find_job_listings_with_gpt()
            
            if gpt_job_listings:
                print(f"✅ Gemini 2.5 Pro found {len(gpt_job_listings)} job listings")
                
                # Show found listings
                print("🏆 Job listings found:")
                for i, listing in enumerate(gpt_job_listings[:self.max_jobs_to_find]):
                    print(f"  {i+1}. '{listing.title}' at {listing.company}")
                
                return gpt_job_listings[:self.max_jobs_to_find]
            else:
                print("❌ No job listings found with Gemini 2.5 Procx")
                return []
            
        except Exception as e:
            print(f"❌ Error finding job listings: {e}")
            # Final fallback to heuristics
            return self.find_job_listings_with_heuristics()
    
    def find_job_listings_with_gpt(self) -> List[JobListing]:
        """Use Gemini 2.5 Pro structured output to find job listings from the entire page source."""
        try:
            from google import genai
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")

            page_source = self.page.content()

            prompt = f"""
                You are analyzing an HTML page that lists multiple job postings, typically shown in a vertical list on the left side.

                Your task is to extract all visible job listings, focusing on these fields:
                    •	title: The visible job title text shown to the user.
                    •	company: The visible company or employer name (return “” if truly absent).
                    •	href: The absolute or relative link to the job details (from the clickable anchor's href).
                    •	selector: A CSS selector that Playwright can use to re-find and click the button or link that opens the job detail page. The selector MUST be stable and as specific as possible (prefer [data-*] attributes, unique IDs, or meaningful classes, and use :nth-of-type if needed).

                Instructions:
                    •	Only include actual job listings, not ads, navigation, or placeholders.
                    •	Listings will often be grouped inside a list (ul, ol, div, etc). Detect jobs from repeating elements.
                    •	The selector must point to the clickable element that opens the job detail page.
                    •	Never include selectors for parent containers, only the clickable item (usually a button or anchor).
                    •	If a job's company is missing, set company to “” (empty string).
                    •	If you are unsure about a field, make the best reasonable guess.

                Return a single JSON object containing job_listings: a list of JobListing objects, like this:
                {{
                    "job_listings": [
                        {{
                        "title": "…",
                        "company": "…",
                        "href": "…",
                        "selector": "…"
                        }},
                        …
                    ]
                }}

                Good Selector Examples:
                    •	'a[data-job-id="1234"]'
                    •	'button.job-card__link'
                    •	'a.job-listing-title'
                    •	'a#job_5678'
                    •	'div.job-card:nth-of-type(2) a'
                    •	'a[href="/jobs/abcd"]'
                    •	'a[data-testid="job-link-5"]'

                Bad Selector Examples:
                    •	'div.job-list'  (parent container, not clickable)
                    •	'ul li'  (too broad, not stable)
                    •	'a'  (too generic, will match all links)
                    •	'.button'  (generic class, may match unrelated buttons)
                    •	'div'  (not clickable)
                    •	'a[href]'  (matches all links, not specific)
                    •	Selectors that match multiple elements without being unique to each job listing

                Good Extraction Example:
                {{
                "job_listings": [
                    {{
                        "title": "Software Engineer",
                        "company": "OpenAI",
                        "href": "/jobs/123",
                        "selector": "a[data-job-id='123']"
                    }},
                    {{
                        "title": "Product Manager",
                        "company": "",
                        "href": "/jobs/124",
                        "selector": "a#job_124"
                    }}
                ]
                }}

                Bad Extraction Example:
                {{
                    "job_listings": [
                    {{
                        "title": "",
                        "company": "OpenAI",
                        "href": "",
                        "selector": "div.job-list"
                    }}
                ]
                }}

                Only return the JSON object. Do not return any other explanation or text.

                HTML Source:
                {page_source}
                """

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert web scraper. Extract job listings from raw HTML and return them.",
                    response_mime_type="application/json",
                    response_schema=JobListingPayload
                )
            )

            payload: JobListingPayload = response.parsed

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

                    title = jl.title.strip()
                    company = jl.company.strip()
                    href = jl.href.strip()

                    if not title or not href:
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
            import traceback
            traceback.print_exc()
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
                            
                            # Vet the job listing with Gemini 2.5 Pro
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
                            
                            # Vet the job listing with Gemini 2.5 Pro
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
        """Use Gemini 2.5 Pro to evaluate if a job is a good fit based on preferences"""
        try:
            from google import genai
            import json as json_module
            
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
            
            # Get Gemini client
            try:
                client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
                
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction="You are an expert job matching assistant. Analyze job descriptions and provide detailed matching scores based on candidate preferences. Return only valid JSON.",
                        response_mime_type="application/json"
                    )
                )
                
                result = response.text.strip()
                
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
                
                print(f"🤖 Gemini 2.5 Pro Evaluation: {evaluation.get('recommendation', 'unknown')} (score: {evaluation.get('overall_score', 'unknown')}/10)")
                print(f"   Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
                
                return evaluation
                
            except Exception as gpt_error:
                print(f"❌ Gemini 2.5 Pro evaluation failed: {gpt_error}")
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
        """Find the apply button on a job page using Gemini 2.5 Pro"""
        try:
            print("🔍 Looking for apply button using Gemini 2.5 Pro...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use Gemini 2.5 Pro to find the apply button
            apply_button_info = self._find_apply_button_with_gpt(page_html)
            
            if apply_button_info:
                # Try to find the actual element using the selector from Gemini
                try:
                    element = self.page.query_selector(apply_button_info['selector'])
                    if element:
                        # Get the href if it's a link
                        href = None
                        try:
                            href = element.get_attribute('href')
                        except:
                            pass
                        
                        print(f"✅ Gemini 2.5 Pro found apply button: '{apply_button_info['text']}' (selector: {apply_button_info['selector']}, href: {href})")
                        return {
                            'text': apply_button_info['text'],
                            'element': element,
                            'selector': apply_button_info['selector'],
                            'href': href
                        }
                except Exception as e:
                    print(f"⚠️ Could not find element with selector '{apply_button_info['selector']}': {e}")
            
            print("❌ Gemini 2.5 Pro could not find apply button")
            return None
            
        except Exception as e:
            print(f"❌ Error finding apply button with Gemini 2.5 Pro: {e}")
            return None
    
    def find_job_listing_url(self) -> Optional[Dict[str, Any]]:
        """Find the URL of a job listing"""
        try:
            # Open the job in another tab, and get the URL, and close the new tab
            apply_button = self.find_apply_button()
            apply_button_selector = apply_button['selector']
            apply_button_element = self.page.query_selector(apply_button_selector)
            apply_button_element.click()
            time.sleep(1)
            url = self.page.url
            self.page.close()
            return url
        except Exception as e:
            print(f"❌ Error finding job listing URL: {e}")
            return None
    
    def _find_apply_button_with_gpt(self, page_html: str) -> Optional[Dict[str, Any]]:
        """Use Gemini 2.5 Pro to find apply button from page HTML"""
        try:
            from google import genai
            import json
            
            # Create the prompt for Gemini 2.5 Pro
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
            
            # Get Gemini client
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Make the API call
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an apply button detection expert. Return only valid JSON with text, selector, and type fields.",
                    response_mime_type="application/json"
                )
            )
            
            # Extract the result
            result = response.text.strip()
            
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
                        print("⚠️ Gemini response missing required fields")
                        return None
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse Gemini response as JSON: {e}")
                return None
                
        except Exception as e:
            print(f"❌ Error in Gemini 2.5 Pro apply button detection: {e}")
            return None
    
    def extract_job_title_from_page(self) -> Optional[str]:
        """Extract job title from the current page using Gemini 2.5 Pro"""
        try:
            print("📋 Extracting job title from current page using Gemini 2.5 Pro...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use Gemini 2.5 Pro to extract the job title
            job_title = self._extract_job_title_with_gpt(page_html)
            
            if job_title and job_title != "Unknown Job Title":
                print(f"✅ Gemini 2.5 Pro extracted job title: '{job_title}'")
                return job_title
            else:
                print("❌ Gemini 2.5 Pro could not extract job title")
                return "Unknown Job Title"
                
        except Exception as e:
            print(f"❌ Error extracting job title with Gemini 2.5 Pro: {e}")
            return "Unknown Job Title"
    
    def _extract_job_title_with_gpt(self, page_html: str) -> str:
        """Use Gemini 2.5 Pro to extract job title from page HTML"""
        try:
            from google import genai
            
            # Create the prompt for Gemini 2.5 Pro
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
            
            # Get Gemini client
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Make the API call
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a job title extraction expert. Return only the job title as a clean string."
                )
            )
            
            # Extract the result
            result = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
            
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
            print(f"❌ Error in Gemini 2.5 Pro job title extraction: {e}")
            return "Unknown Job Title"
    
    def extract_job_description_from_page(self) -> Optional[str]:
        """Extract job description from the current page using Gemini 2.5 Pro"""
        try:
            print("📄 Extracting job description from current page using Gemini 2.5 Pro...")
            
            # Get the page HTML content
            page_html = self.page.content()
            
            # Use Gemini 2.5 Pro to extract the job description
            job_description = self._extract_job_description_with_gpt(page_html)
            
            if job_description:
                print(f"✅ Gemini 2.5 Pro extracted job description ({len(job_description)} characters)")
                return job_description
            else:
                print("❌ Gemini 2.5 Pro could not extract job description")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting job description with Gemini 2.5 Pro: {e}")
            return None
    
    def _extract_job_description_with_gpt(self, page_html: str) -> Optional[str]:
        """Use Gemini 2.5 Pro to extract job description from page HTML"""
        try:
            from google import genai
            
            # Create the prompt for Gemini 2.5 Pro
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
            
            # Get Gemini client
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Make the API call
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a job description extraction expert. Return only the job description as clean, well-formatted text."
                )
            )
            
            # Extract the result
            result = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
            
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
            print(f"❌ Error in Gemini 2.5 Pro job description extraction: {e}")
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
             
    def run_bot(self, job_site_url: str, transition_handler: JobDetailToFormTransitionHandler = None) -> bool:
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
        """
        print(f"🚀 Starting job application for: {job_site_url}")
        
        try:
            # Initialize browser and navigate to the job site
            if not self.start_browser():
                return False
            if not self.open_url(job_site_url):
                return False
            
            self.transition_handler = transition_handler
            
            # Main automation loop - runs until completion or error
            max_iterations = 50  # Safety limit to prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                self.click_accept_cookies_button()
                
                iteration += 1
                print(f"\n🔄 Iteration {iteration}/{max_iterations}")
                                    
                # Detect the page type and handle it directly
                # This uses the page detector to analyze the current page
                page_type_result = self.handle_page_type()
                if page_type_result == PageTypeResult.SUCCESS:
                    print("✅ Done with Finding Jobs -> Navigating to Form -> Filling Form")
                    
                    return True
                elif page_type_result == PageTypeResult.ERROR:
                    return False
            
            print("⚠️ Max iterations reached")
            return False
            
        except Exception as e:
            print(f"❌ Error in job application: {e}")
            return False
        finally:
            if self.page:
                self.page.pause()
            else:
                print("⚠️ Browser not started, skipping pause")
    
    def _fill_search_form(self) -> TextApplicationFieldResponse:
        """Fill the job search form with preferences"""
        try:
            print("🔍 Finding relevant fields for search form...")
            search_inputs = self.find_search_page_input_fields()
            
            if not search_inputs:
                print("❌ No relevant fields found")
                return False
            
            print(f"🔍 Found {len(search_inputs.fields)} relevant fields")
            
            fill_result = self.fill_text_input_fields(search_inputs)
            if not fill_result:
                print("❌ Failed to fill text input fields")
                return False
            
            print(f"✅ Filled {len(search_inputs.fields)} fields with job preferences")
            
            # Submit the form
            submit_button = self.find_search_page_submit_button()
            if submit_button:
                try:
                    submit_button_element = self.page.locator(submit_button.field_selector)
                    submit_button_element.click()
                    time.sleep(3)
                    print(f"✅ Successfully clicked submit button: '{submit_button.field_name}'")
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

                    job_element.click()
                    time.sleep(3)  # Wait for job details to load
                except Exception as e:
                    print(f"❌ Error clicking job listing: {e}")
                    continue
                
                # Check if we're now on a job detail page (either full page or panel)
                print("🔍 Checking if job details loaded...")
                
                print("✅ Proceeding with evaluation")
                # The job detail page handler will evaluate and open applications in new tabs
                self._handle_job_detail_page(job_listing)
                
                self.clear_modal_if_present(job_listing.title)
                
            return True
        except Exception as e:
            print(f"❌ Error processing job listings: {e}")
            return False
    
    def clear_modal_if_present(self, modal_title: str = None, max_retries: int = 3):
        """Clear any modal if present with retry logic for API errors"""
        import time
        
        print("🔍 Checking for modal")
        
        for attempt in range(max_retries):
            try:
                
                client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
                prompt = f"""
                Analyze the provided HTML of the current webpage and identify if there is a modal dialog currently present and visible to the user.  
                A modal is a popup, dialog, overlay, or window that prevents interaction with the main page until it is dismissed (e.g., login dialogs, cookie consent popups, alert overlays).

                Your objectives:
                1. Determine if a modal dialog is currently visible.
                2. If so, identify the single button, link, or element that closes the modal when clicked (e.g., "Close", "X", "Dismiss", "Cancel", etc).
                3. Return ONLY the most specific, stable, and reliable CSS selector for the close button.

                Guidelines for selector:
                - The close button will be in a modal with title {modal_title}. Ensure that the selector is with the modal title.
                - Selector must be valid, stable, and as specific as possible (use data-* attributes, IDs, unique classes, nth-of-type, etc).
                - Selector must directly target the clickable close element (never a parent container, never a generic button).
                - Do NOT use extremely generic selectors (like "button" or ".btn").
                - Do NOT return XPath or non-CSS selectors.
                - Do NOT include anything except the selector string in the response.
                - If no modal is present, return an empty string ("").
                - The close button is often an X icon or a close button in the modal

                Good selector examples:
                "button[data-testid='modal-close']"
                "button.close-modal"
                "div[role='dialog'] button[aria-label='Close']"
                "button#closeDialogButton"
                "a.modal__close"
                "span.icon-close"
                "button[class*='close']"
                "button[aria-label='Dismiss']"
                "div.cookie-modal button:nth-of-type(2)"

                Bad selector examples (do NOT return):
                "button"
                ".btn"
                ".close"      (if not unique on the page)
                "div"         (never target a non-clickable container)
                ".modal"      (the modal, not the button)
                "input"       (unless it's actually the close control)
                Any XPath selector
                Anything that matches multiple, unrelated elements

                Your output should be:
                - If a modal is present: the CSS selector for the close button, nothing else.
                - If no modal is present: return "" (an empty string).

                HTML of the page:
                {self.page.content()}
                """

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction="You are an expert at analyzing web pages and identifying modal elements.",
                    )
                )
                
                if response.text:
                    print(f"🔍 Attempting to clear modal (attempt {attempt + 1}/{max_retries})...")
                    
                    close_button_selector = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
                    close_button = self.page.query_selector(close_button_selector)
                    if close_button:
                        print(f"✅ Found close button: {close_button_selector}")
                        close_button.click()
                        return True
                    else:
                        # Try and click outside the modal
                        print("❌ No close button found, clicking outside the modal")
                        self.page.click("body")
                        return True
                else:
                    print("ℹ️ No modal detected")
                    return True

            except Exception as e:
                error_message = str(e)
                print(f"❌ Error clearing modal (attempt {attempt + 1}/{max_retries}): {error_message}")
                
                # Check if it's a 500 internal error or other retryable error
                if "500 INTERNAL" in error_message or "internal error" in error_message.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"🔄 Retrying in {wait_time} seconds due to internal server error...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ Max retries reached for internal server error")
                        return self._fallback_modal_clear()
                
                # Check if it's an API quota or rate limit error
                elif any(keyword in error_message.lower() for keyword in ["quota", "rate limit", "too many requests", "429"]):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Longer wait for rate limits: 5s, 10s, 15s
                        print(f"🔄 Retrying in {wait_time} seconds due to rate limit...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ Max retries reached for rate limit")
                        return self._fallback_modal_clear()
                
                # For other errors, don't retry
                else:
                    print("❌ Non-retryable error encountered")
                    return self._fallback_modal_clear()
        
        # If we get here, all retries failed
        print("❌ All retry attempts failed")
        return self._fallback_modal_clear()
    
    def _fallback_modal_clear(self) -> bool:
        """Fallback method to clear modals when API fails"""
        try:
            print("🔄 Using fallback modal clearing method...")
            
            # Common modal close selectors to try
            common_close_selectors = [
                'button[aria-label*="Close"]',
                'button[aria-label*="Dismiss"]',
                'button[title*="Close"]',
                'button[title*="Dismiss"]',
                '.modal-close',
                '.modal__close',
                '.close-modal',
                '.dialog-close',
                '.popup-close',
                'button.close',
                'a.close',
                'span.close',
                'button[class*="close"]',
                'button[class*="dismiss"]',
                'button[class*="cancel"]',
                '.btn-close',
                '.btn-dismiss',
                '.btn-cancel',
                '[data-testid*="close"]',
                '[data-testid*="dismiss"]',
                'button:has-text("Close")',
                'button:has-text("Dismiss")',
                'button:has-text("Cancel")',
                'button:has-text("×")',
                'button:has-text("X")',
                'span:has-text("×")',
                'span:has-text("X")',
                'a:has-text("Close")',
                'a:has-text("Dismiss")'
            ]
            
            # Try each selector
            for selector in common_close_selectors:
                try:
                    element = self.page.query_selector(selector)
                    if element and element.is_visible():
                        print(f"✅ Found fallback close button: {selector}")
                        element.click()
                        time.sleep(1)  # Wait for modal to close
                        return True
                except Exception as e:
                    continue
            
            # If no close button found, try clicking outside the modal
            print("🔄 No close button found, trying to click outside modal...")
            try:
                # Click in the top-left corner of the page (usually outside modals)
                self.page.click("body", position={"x": 10, "y": 10})
                time.sleep(1)
                
                # Also try pressing Escape key
                self.page.keyboard.press("Escape")
                time.sleep(1)
                
                print("✅ Fallback modal clearing completed")
                return True
                
            except Exception as e:
                print(f"❌ Fallback modal clearing failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error in fallback modal clearing: {e}")
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

    def handle_page_type(self) -> PageTypeResult:
        """Detect current page type and handle it appropriately"""
        if not self.page_detector:
            print("❌ Page detector not initialized")
            return PageTypeResult.ERROR
        
        # Detect page type
        page_type = self.page_detector.detect_page_type()
        
        # Handle different page types
        if page_type == PageType.SEARCH_PAGE:
            return self._handle_search_page()
        elif page_type == PageType.JOB_DETAIL_PAGE:
            return self._handle_job_detail_page()
        elif page_type == PageType.APPLICATION_PAGE:
            return self._handle_application_page()
        elif page_type == PageType.LOGIN_PAGE:
            return self._handle_login_page()
        elif page_type == PageType.CAPTCHA_PAGE:
            return self._handle_captcha_page()
        elif page_type == PageType.ERROR_PAGE:
            return self._handle_error_page()
        elif page_type == PageType.RESULTS_PAGE:
            self._handle_search_results_page()
            
            if len(self.rejected_job_listings) > 0:
                original_page = self.page
                context = self.page.context
                
                for job_listing in self.rejected_job_listings:
                    # Duplocate the tab
                    print(f"Duplicating tab for {job_listing.job_listing.title}")
                    with context.expect_page() as new_page_info:
                        original_page.evaluate("window.open()")
                    new_page = new_page_info.value
                    
                    try:
                        new_page.goto(original_page.url, wait_until="domcontentloaded")
                    except Exception as e:
                        print(f"⚠️ Could not duplicate URL into new tab: {e}")

                    self.page = new_page
                    print(f"Navigating to the job form for {job_listing.job_listing.title}")
                    
                    try:
                        if not self._click_job_listing(job_listing.job_listing):
                            print(f"❌ Failed to click job listing: {job_listing.job_listing.title}")
                            continue
                        
                        handler = self.transition_handler or JobDetailToFormTransitionHandler(new_page)
                        navigation_result = handler.navigate_to_form()
                        
                        if navigation_result.success:
                            if navigation_result.requires_user_intervention:
                                print(f"⏸️ User intervention required for {job_listing.job_listing.title}")
                                new_page.pause()
                                continue
                            elif navigation_result.form_ready:
                                print(f"✅ Successfully navigated to job form for {job_listing.job_listing.title}")
                                
                                # Fill the form
                                app_filler = ApplicationFiller(self.page, self.preferences)
                                app_filler.fill_application(
                                    on_success_callback=lambda: print(f"✅ Application filling completed successfully for {job_listing.job_listing.title}"),
                                    on_failure_callback=lambda: print(f"❌ Application filling failed for {job_listing.job_listing.title}")
                                )
                                
                                # Go back to the original tab
                        else:
                            print(f"❌ Failed to navigate to job form: {job_listing.job_listing.title}")
                            continue
                        
                    finally:
                        # Close the new page
                        new_page.close()
                        
                        self.page = original_page
                
                return PageTypeResult.SUCCESS
            else:
                print("❌ No accepted job listings found")
                return PageTypeResult.SUCCESS
                
                
                
        else:
            return self._handle_unknown_page()
    
    def _click_job_listing(self, job_listing: JobListing) -> bool:
        """Click on a job listing"""
        try:
            # self.clear_modal_if_present(job_listing.title)
            job_element = self.page.query_selector(job_listing.selector)
            job_element.click()
            time.sleep(3)
            return True
        except Exception as e:
            print(f"❌ Error clicking job listing: {e}")
            return False
    
    def _handle_search_page(self) -> PageTypeResult:
        """Handle job search form page"""
        print("🔍 Handling search page...")
        
        # Wait for the user to complete any login process needed
        # print("⏸️ Waiting for user to complete login process if needed...")
        # self.page.pause()
        
        # Resume when user is ready
        print("📝 Filling search form...")
        success = self._fill_search_form()
        if not success:
            print("❌ Failed to fill search form")
            return PageTypeResult.ERROR
        
        return PageTypeResult.CONTINUE
    
    def _handle_search_results_page(self) -> PageTypeResult:
        """Handle job search results page"""
        print("📋 Handling results page...")
        
        job_listings = self.find_job_listings()
        if job_listings:
            print(f"✅ Found {len(job_listings)} job listings")
            # Process jobs
            success = self._process_job_listings(job_listings)
            if not success:
                print("❌ Failed to process job listings")
                return PageTypeResult.ERROR
            
            print(f"✅ Accepted {len(self.accepted_job_listings)} job listings")
            print(f"❌ Rejected {len(self.rejected_job_listings)} job listings")
            
        else:
            print("⚠️ No job listings found")
            return False
        
        return PageTypeResult.SUCCESS
    
    def _handle_job_detail_page(self, job_listing: JobListing) -> PageTypeResult:
        """Handle individual job posting page"""
        print("📄 Handling job detail page...")
        
        try:
            # Extract job description from the page
            print("🔍 Extracting job description...")
            job_title = self.extract_job_title_from_page()
            job_description = self.extract_job_description_from_page()
            
            if not job_description:
                print("❌ Could not extract job description")
                return PageTypeResult.ERROR
            
            print(f"✅ Extracted job description ({len(job_description)} characters)")
            
            print(f"📋 Job: {job_title}")
            
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
                self.accepted_job_listings.append(JobListingDetails(
                    job_listing=job_listing,
                    job_description=job_description,
                    job_fit=evaluation,
                    job_url=""
                ))
                return PageTypeResult.CONTINUE
            else:
                print("❌ Job doesn't match preferences well enough")
                self.rejected_job_listings.append(JobListingDetails(
                    job_listing=job_listing,
                    job_description=job_description,
                    job_fit=evaluation,
                    job_url=""
                ))
                return PageTypeResult.ERROR
        except Exception as e:
            print(f"❌ Error handling job detail page: {e}")
            return PageTypeResult.ERROR
          
    def _handle_application_page(self) -> PageTypeResult:
        """Handle job application form page"""
        print("📝 Handling application page...")
        
        print("📝 Filling application form...")
        success = self._fill_application_form()
        if not success:
            print("❌ Failed to fill application form")
            return PageTypeResult.ERROR
        
        # After filling form, we expect navigation to next step
        return PageTypeResult.CONTINUE
    
    def _handle_login_page(self) -> PageTypeResult:
        """Handle login/authentication page"""
        print("🔒 Handling login page...")
        print("⏸️ Pausing for user login. Please log in manually and resume.")
        self.page.pause()
        
        # After resume, check if login was successful
        time.sleep(2)
        new_detection = self.page_detector.detect_page_type()
        if new_detection == PageType.LOGIN_PAGE:
            print("❌ Still on login page. Please complete login and resume again.")
            self.page.pause()
        
        print("✅ Login appears complete. Continuing...")
        return PageTypeResult.CONTINUE
    
    def _handle_captcha_page(self) -> PageTypeResult:
        """Handle CAPTCHA verification page"""
        print("🤖 Handling CAPTCHA page...")
        print("⏸️ CAPTCHA detected. Please solve the CAPTCHA manually and resume.")
        self.page.pause()
        
        # After resume, check if CAPTCHA was solved
        time.sleep(2)
        new_detection = self.page_detector.detect_page_type()
        if new_detection == PageType.CAPTCHA_PAGE:
            print("❌ Still on CAPTCHA page. Please solve the CAPTCHA and resume again.")
            self.page.pause()
        
        print("✅ CAPTCHA appears solved. Continuing...")
        return {'action': None, 'page_type': 'captcha_page', 'reason': 'captcha_solved'}
    
    def _handle_error_page(self) -> PageTypeResult:
        """Handle error pages"""
        print("❌ Handling error page...")
        return PageTypeResult.ERROR
    
    def _handle_unknown_page(self) -> PageTypeResult:
        """Handle unknown page types"""
        print("❓ Handling unknown page...")
        
        # Wait for user intervention
        print("⏸️ Waiting for user intervention...")
        self.page.pause()
        return PageTypeResult.ERROR

    def vet_job_listing_with_gpt(self, job_title: str, company_name: str = "", href: str = "") -> bool:
        """Use Gemini 2.5 Pro to determine if this is actually a job listing"""
        try:
            from google import genai
            
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
                client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
                
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction="You are a job listing classifier. Determine if text represents an actual job posting or just navigation/UI elements. Return only YES or NO."
                    )
                )
                
                result = response.text.strip().upper()
                
                is_job = result == "YES"
                print(f"🤖 Gemini 2.5 Pro vetting: '{job_title}' -> {result} ({'Job' if is_job else 'Not a job'})")
                
                return is_job
                
            except Exception as gpt_error:
                print(f"❌ Gemini 2.5 Pro vetting failed: {gpt_error}")
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
     
    def click_accept_cookies_button(self) -> bool:
        """
        Click the accept cookies button
        
        Returns:
            bool: True if button was found and clicked successfully
        """
        try:
            print("🔍 Looking for accept cookies button...")
            
            # Common accept cookies button selectors
            accept_cookies_selectors = [
                'button:has-text("Accept all")',
                'button:has-text("Accept cookies")',
                'button:has-text("Accept")',
                'button:has-text("Allow all")',
                'button:has-text("Allow")',
                'button:has-text("Allow cookies")',
            ]
            
            accept_cookies_button = None
            for selector in accept_cookies_selectors:
                try:
                    accept_cookies_button = self.page.locator(selector)
                    if accept_cookies_button and accept_cookies_button.is_visible():
                        print(f"✅ Found accept cookies button with selector: {selector}")
                        break
                except:
                    continue
            
            if not accept_cookies_button:
                print("ℹ️ No accept cookies button found")
                return False
            
            # Click the accept cookies button
            print("🎯 Clicking accept cookies button...")
            accept_cookies_button.click()
            
            return True
        
        except Exception as e:
            print(f"❌ Error clicking accept cookies button: {e}")
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
    
    bot = FindJobsBot(headless=False, preferences=preferences)
    
    # Test the bot
    print("🧪 Testing Hybrid Browser Bot")
    print("=" * 50)
    
    success = bot.run_bot("https://www.reed.co.uk/")
    
    if success:
        print("✅ Job application test completed successfully!")
    else:
        print("❌ Job application test failed") 