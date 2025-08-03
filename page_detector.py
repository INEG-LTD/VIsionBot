#!/usr/bin/env python3
"""
Page Detector Module

This module detects what type of page we're on and determines the best action to take.
"""

from typing import Dict, Any
from enum import Enum
from playwright.sync_api import Page

class PageType(Enum):
    """Enumeration of different page types"""
    SEARCH_PAGE = "search_page"           # Job search form page
    RESULTS_PAGE = "results_page"         # Job search results page
    JOB_DETAIL_PAGE = "job_detail_page"   # Individual job posting page
    APPLICATION_PAGE = "application_page" # Job application form page
    LOGIN_PAGE = "login_page"             # Login/authentication page
    CAPTCHA_PAGE = "captcha_page"         # CAPTCHA verification page
    ERROR_PAGE = "error_page"             # Error page (404, 500, etc.)
    UNKNOWN_PAGE = "unknown_page"         # Unknown page type

    @classmethod
    def get_page_type(cls, page_type: str) -> "PageType":
        if page_type == "search_page":
            return cls.SEARCH_PAGE
        elif page_type == "results_page":
            return cls.RESULTS_PAGE
        elif page_type == "job_detail_page":
            return cls.JOB_DETAIL_PAGE
        elif page_type == "application_page":
            return cls.APPLICATION_PAGE
        elif page_type == "login_page":
            return cls.LOGIN_PAGE
        elif page_type == "captcha_page":
            return cls.CAPTCHA_PAGE
        elif page_type == "error_page":
            return cls.ERROR_PAGE
        else:
            return cls.UNKNOWN_PAGE

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
    
    def __hash__(self):
        return hash(self.value)
    
    
    
class PageDetector:
    def __init__(self, page: Page):
        self.page = page
        self._cache = {}  # Cache for detection results
        self._last_url = None
        self._last_detection = None
    
    def detect_page_type(self) -> PageType:
        """Detect the type of the current page using multiple strategies"""
        try:
            current_url = self.page.url
            
            print("🔍 Detecting page type...")
            print(f"📍 URL: {current_url}")
            
            title = self.page.title()
            print(f"📄 Title: {title}")
            
            # Try detection methods in order of accuracy (most accurate first)
            detection_result = PageType.UNKNOWN_PAGE

            print("🔍 Detecting page type...")
            try:
                gemini_result = self._detect_with_gemini(current_url, title)
                if gemini_result != PageType.UNKNOWN_PAGE:
                    detection_result = gemini_result    
                    print(f"  ✅ Gemini detection successful! {gemini_result.value}")
                else:
                    print("  ⚠️ Gemini detection and page type detection failed")
            except Exception as e:
                print(f"  ❌ Error in Gemini detection: {e}")
            
            # Cache the result
            self._last_url = current_url
            
            print(f"✅ Page type detected: {detection_result.value}")
            
            return detection_result
            
        except Exception as e:
            print(f"❌ Error detecting page type: {e}")
            return PageType.UNKNOWN_PAGE
    
    def _detect_with_gemini(self, page_url: str, title: str) -> PageType:
        """Use Gemini 2.5 Pro to analyze page type based on URL and title"""
        try:
            from google import genai
            
            # Get some page content for context
            page_content = self.page.content()
            
            prompt = f"""
                You are an expert page-classifier. Read the supplied metadata, ignore all HTML markup, and return **one** label—exactly as written— matching the page.

                ### Input
                • URL: {page_url}  
                • Title: {title}  
                • Content snippet (truncated): {page_content}

                ### Allowed output labels
                search_page            - Job-search form where users enter keywords, location, or filters  
                results_page           - List of multiple job postings (may span pages)  
                job_detail_page        - Single job posting with full description / “Apply” button  
                application_page       - Form that collects applicant data (name, CV upload, etc.)  
                login_page             - Authentication page (login, register, forgot-password)  
                captcha_page           - CAPTCHA or bot-verification interstitial  
                error_page             - HTTP or application error (404, 403, 500, maintenance)  
                company_profile_page   - Employer overview (about, benefits, culture)  
                static_info_page       - Non-job static content (blog article, help/FAQ, contact, T&C)  
                unknown_page           - Doesn't fit any label or information is insufficient

                ### Output contract
                Return **only** the label on its own line with no quotes, no extra whitespace, and no additional text.

                Examples  
                search_page  
                results_page  
                job_detail_page  
                application_page  
                login_page  
                captcha_page  
                error_page  
                company_profile_page  
                static_info_page  
                unknown_page
                """
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing webpages and determining page types for job search automation. Return only valid JSON.",
                    response_mime_type="application/json",
                    response_schema=PageType
                )
            )
            
            result = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
            
            return PageType.get_page_type(result)
            
        except Exception as e:
            print(f"❌ Error in Gemini detection: {e}")
            return PageType.UNKNOWN_PAGE
    
    def _get_recommended_action(self, page_type: PageType) -> str:
        """Determine the recommended action based on page type"""
        action_map = {
            PageType.SEARCH_PAGE: 'fill_search_form',
            PageType.RESULTS_PAGE: 'find_job_listings',
            PageType.JOB_DETAIL_PAGE: 'evaluate_and_apply',
            PageType.APPLICATION_PAGE: 'fill_application_form',
            PageType.LOGIN_PAGE: 'wait_for_user_login',
            PageType.CAPTCHA_PAGE: 'wait_for_user_captcha',
            PageType.ERROR_PAGE: 'handle_error',
            PageType.UNKNOWN_PAGE: 'wait_for_user'
        }
        
        return action_map.get(page_type, 'wait_for_user')
    
    def should_restart(self) -> bool:
        """Determine if the program should restart based on current page"""
        # For now, restart if we're on an unknown page with low confidence
        current_detection = self.detect_page_type()
        return current_detection == PageType.UNKNOWN_PAGE
    
 