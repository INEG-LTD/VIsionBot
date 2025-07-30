#!/usr/bin/env python3
"""
Page Detector Module

This module detects what type of page we're on and determines the best action to take.
Uses a combination of URL analysis, content analysis, and GPT-4o image recognition.
"""

import time
import re
from typing import Dict, Any, Optional, List
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

class PageDetector:
    def __init__(self, page: Page):
        self.page = page
        self._cache = {}  # Cache for detection results
        self._last_url = None
        self._last_detection = None
    
    def detect_page_type(self) -> Dict[str, Any]:
        """Detect the type of the current page using multiple strategies"""
        try:
            current_url = self.page.url
            
            # Check cache first
            # if current_url == self._last_url and self._last_detection:
            #     print("  📋 Using cached detection result")
            #     return self._last_detection
            
            print("🔍 Detecting page type...")
            print(f"📍 URL: {current_url}")
            
            title = self.page.title()
            print(f"📄 Title: {title}")
            
            # Try detection methods in order of accuracy (most accurate first)
            detection_result = None
            detection_method = 'fallback'
            
            # 1. Image-based detection (most accurate - if OpenAI API is available)
            print("🔍 Trying image-based detection...")
            try:
                image_result = self._detect_by_image()
                if image_result and image_result.get('confidence', 0) > 0.7:
                    detection_result = image_result
                    detection_method = 'image_recognition'
                    print("  ✅ Image detection successful!")
                else:
                    print("  ⚠️ Image detection failed or low confidence, trying content-based...")
            except Exception as e:
                print(f"  ❌ Error in image detection: {e}")
                print("  🔄 Falling back to content-based detection...")
            
            # 2. Content-based detection (if image detection failed)
            if not detection_result:
                print("🔍 Trying content-based detection...")
                try:
                    content_result = self._detect_by_content()
                    if content_result and content_result.get('confidence', 0) > 0.6:
                        detection_result = content_result
                        detection_method = 'content_analysis'
                        print("  ✅ Content detection successful!")
                    else:
                        print("  ⚠️ Content detection failed or low confidence, trying URL-based...")
                except Exception as e:
                    print(f"  ❌ Content detection failed: {e}")
                    print("  🔄 Falling back to URL-based detection...")
            
            # 3. URL-based detection (if content detection failed)
            if not detection_result:
                print("🔍 Trying URL-based detection...")
                try:
                    url_result = self._detect_by_url(current_url)
                    if url_result and url_result.get('confidence', 0) > 0.5:
                        detection_result = url_result
                        detection_method = 'url_analysis'
                        print("  ✅ URL detection successful!")
                    else:
                        print("  ⚠️ URL detection failed or low confidence")
                except Exception as e:
                    print(f"  ❌ URL detection failed: {e}")
            
            # 4. GPT-based detection (if all others failed)
            if not detection_result:
                print("🔍 Trying GPT-based detection...")
                try:
                    gpt_result = self._detect_with_gpt(current_url, title)
                    if gpt_result and gpt_result.get('confidence', 0) > 0.5:
                        detection_result = gpt_result
                        detection_method = 'gpt_analysis'
                        print("  ✅ GPT detection successful!")
                    else:
                        print("  ⚠️ GPT detection failed or low confidence")
                except Exception as e:
                    print(f"  ❌ Error in GPT detection: {e}")
            
            # Determine final result
            if detection_result:
                page_type = detection_result['page_type']
                confidence = detection_result.get('confidence', 0.5)
            else:
                # Fallback to unknown page type
                page_type = PageType.UNKNOWN_PAGE
                confidence = 0.1
                detection_method = 'fallback'
            
            # Get recommended action
            recommended_action = self._get_recommended_action(page_type)
            
            result = {
                'page_type': page_type,
                'confidence': confidence,
                'detection_method': detection_method,
                'recommended_action': recommended_action,
                'url': current_url,
                'title': title
            }
            
            # Cache the result
            self._last_url = current_url
            self._last_detection = result
            
            print(f"✅ Page type detected: {page_type.value}")
            print(f"🎯 Recommended action: {recommended_action}")
            print(f"📊 Confidence: {confidence:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error detecting page type: {e}")
            return {
                'page_type': PageType.UNKNOWN_PAGE,
                'confidence': 0.0,
                'detection_method': 'error',
                'recommended_action': 'handle_error',
                'url': self.page.url if hasattr(self, 'page') else 'unknown',
                'title': 'unknown'
            }
    

    
    def _detect_by_url(self, page_url: str) -> Dict[str, Any]:
        """Detect page type based on URL patterns"""
        url_lower = page_url.lower()
        
        # Search page patterns
        search_patterns = [
            r'/jobs\?', r'/search\?', r'/find-jobs', r'/job-search',
            r'q=', r'keywords=', r'search=', r'query='
        ]
        
        # Results page patterns
        results_patterns = [
            r'/jobs$', r'/search$', r'/results', r'/listings',
            r'jobs/search', r'search/jobs'
        ]
        
        # Job detail patterns
        detail_patterns = [
            r'/job/', r'/position/', r'/career/', r'/viewjob',
            r'job-details', r'job-description', r'apply'
        ]
        
        # Login patterns
        login_patterns = [
            r'/login', r'/signin', r'/sign-in', r'/auth',
            r'/account', r'/profile', r'/dashboard'
        ]
        
        # CAPTCHA patterns
        captcha_patterns = [
            r'captcha', r'verify', r'robot', r'security',
            r'challenge', r'recaptcha'
        ]
        
        # Check patterns
        if any(re.search(pattern, url_lower) for pattern in captcha_patterns):
            return {
                'page_type': PageType.CAPTCHA_PAGE,
                'confidence': 0.9,
                'method': 'url_analysis',
                'page_indicators': ['captcha_url_pattern']
            }
        
        if any(re.search(pattern, url_lower) for pattern in login_patterns):
            return {
                'page_type': PageType.LOGIN_PAGE,
                'confidence': 0.85,
                'method': 'url_analysis',
                'page_indicators': ['login_url_pattern']
            }
        
        if any(re.search(pattern, url_lower) for pattern in detail_patterns):
            return {
                'page_type': PageType.JOB_DETAIL_PAGE,
                'confidence': 0.8,
                'method': 'url_analysis',
                'page_indicators': ['job_detail_url_pattern']
            }
        
        if any(re.search(pattern, url_lower) for pattern in search_patterns):
            return {
                'page_type': PageType.SEARCH_PAGE,
                'confidence': 0.75,
                'method': 'url_analysis',
                'page_indicators': ['search_url_pattern']
            }
        
        if any(re.search(pattern, url_lower) for pattern in results_patterns):
            return {
                'page_type': PageType.RESULTS_PAGE,
                'confidence': 0.7,
                'method': 'url_analysis',
                'page_indicators': ['results_url_pattern']
            }
        
        return {
            'page_type': PageType.UNKNOWN_PAGE,
            'confidence': 0.0,
            'method': 'url_analysis',
            'page_indicators': []
        }
    
    def _detect_by_content(self) -> Dict[str, Any]:
        """Detect page type based on page content and elements"""
        try:
            print("  📄 Getting page content...")
            # Get page content
            body_element = self.page.query_selector('body')
            print(f"  📄 Body element found: {body_element is not None}")
            if body_element:
                print("  📄 Getting inner text...")
                body_text = body_element.inner_text()
                print(f"  📄 Inner text type: {type(body_text)}")
                if body_text:
                    print("  📄 Converting to lowercase...")
                    body_text = body_text.lower()
                else:
                    body_text = ""
            else:
                body_text = ""
            print(f"  📄 Final body text length: {len(body_text)}")
            
            # Check for specific elements
            search_form = self.page.query_selector('form[action*="search"], form[action*="jobs"], input[name*="q"], input[name*="keywords"]')
            job_listings = self.page.query_selector_all('[class*="job"], [class*="position"], [class*="listing"]')
            apply_button = self.page.query_selector('button:has-text("Apply"), a:has-text("Apply"), [class*="apply"]')
            login_form = self.page.query_selector('form[action*="login"], form[action*="signin"], input[name="email"], input[name="username"]')
            captcha_element = self.page.query_selector('[class*="captcha"], [class*="recaptcha"], iframe[src*="recaptcha"]')
            
            indicators = []
            confidence = 0.0
            page_type = PageType.UNKNOWN_PAGE
            
            # CAPTCHA detection
            if captcha_element or 'captcha' in body_text or 'verify you are human' in body_text:
                page_type = PageType.CAPTCHA_PAGE
                confidence = 0.95
                indicators.append('captcha_element')
            
            # Login page detection
            elif login_form or 'sign in' in body_text or 'log in' in body_text:
                page_type = PageType.LOGIN_PAGE
                confidence = 0.9
                indicators.append('login_form')
            
            # Application page detection
            elif apply_button or 'application' in body_text or 'apply for this position' in body_text:
                page_type = PageType.APPLICATION_PAGE
                confidence = 0.85
                indicators.append('apply_button')
            
            # Job detail page detection (including panels)
            job_detail_indicators = [
                'job description', 'requirements', 'responsibilities', 'qualifications',
                'about the role', 'what you\'ll do', 'what we\'re looking for',
                'key responsibilities', 'essential skills', 'preferred qualifications',
                'benefits', 'compensation', 'salary', 'location', 'remote work',
                'full-time', 'part-time', 'contract', 'permanent'
            ]
            
            # Check if we have job detail content
            has_job_detail_content = any(indicator in body_text for indicator in job_detail_indicators)
            
            # Also check for job detail specific elements
            job_detail_elements = self.page.query_selector_all('[class*="job-detail"], [class*="job-description"], [class*="job-info"], [class*="position-detail"]')
            
            if has_job_detail_content or len(job_detail_elements) > 0:
                page_type = PageType.JOB_DETAIL_PAGE
                confidence = 0.85 if has_job_detail_content and len(job_detail_elements) > 0 else 0.75
                indicators.append('job_description_content')
                if len(job_detail_elements) > 0:
                    indicators.append('job_detail_elements')
            
            # Search results page detection (but check if job details are also present)
            has_job_listings = len(job_listings) > 0 or 'jobs found' in body_text or 'search results' in body_text
            
            if has_job_listings:
                # If we have both job listings AND job detail content, it's likely a results page with a job detail panel
                if has_job_detail_content:
                    # This is a results page with a job detail panel - prioritize job detail detection
                    page_type = PageType.JOB_DETAIL_PAGE
                    confidence = 0.8
                    indicators.append('job_listings_with_detail_panel')
                else:
                    # This is a regular results page
                    page_type = PageType.RESULTS_PAGE
                    confidence = 0.75
                    indicators.append('job_listings')
            
            # Search page detection
            elif search_form or 'search for jobs' in body_text or 'find your next job' in body_text:
                page_type = PageType.SEARCH_PAGE
                confidence = 0.7
                indicators.append('search_form')
            
            return {
                'page_type': page_type,
                'confidence': confidence,
                'method': 'content_analysis',
                'page_indicators': indicators
            }
            
        except Exception as e:
            print(f"❌ Error in content detection: {e}")
            return {
                'page_type': PageType.UNKNOWN_PAGE,
                'confidence': 0.0,
                'page_indicators': []
            }
    
    def _detect_by_image(self) -> Dict[str, Any]:
        """Detect page type using GPT-4o image recognition"""
        try:
            # Take a screenshot
            screenshot = self.page.screenshot(type='jpeg', quality=80)
            
            # Use GPT-4o to analyze the image
            import openai
            import base64
            import json as json_module
            
            # Convert screenshot to base64
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            prompt = """
            Analyze this webpage screenshot and determine what type of page it is.
            
            Possible page types:
            1. SEARCH_PAGE - Job search form with input fields for keywords, location, etc.
            2. RESULTS_PAGE - List of job listings/cards showing multiple job opportunities
            3. JOB_DETAIL_PAGE - Detailed view of a single job posting with description
            4. APPLICATION_PAGE - Job application form with fields to fill out
            5. LOGIN_PAGE - Login/sign-in form with email/password fields
            6. CAPTCHA_PAGE - CAPTCHA verification page with puzzle/challenge
            7. ERROR_PAGE - Error page (404, 500, etc.)
            
            Return a JSON response with:
            {
                "page_type": "SEARCH_PAGE|RESULTS_PAGE|JOB_DETAIL_PAGE|APPLICATION_PAGE|LOGIN_PAGE|CAPTCHA_PAGE|ERROR_PAGE",
                "confidence": 0.0-1.0,
                "reasoning": "Brief explanation of why you think this is the page type",
                "key_elements": ["list", "of", "key", "elements", "you", "see"]
            }
            """
            
            client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing webpage screenshots and identifying page types."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}}
                    ]}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate JSON response
            if not result:
                print("  ❌ Empty response from GPT-4o")
                raise Exception("Empty response from GPT-4o")
            
            # Clean up markdown code blocks if present
            if result.startswith('```json'):
                result = result[7:]  # Remove ```json
            if result.startswith('```'):
                result = result[3:]   # Remove ```
            if result.endswith('```'):
                result = result[:-3]  # Remove trailing ```
            
            result = result.strip()
            
            try:
                analysis = json_module.loads(result)
            except json_module.JSONDecodeError as json_error:
                print(f"  ❌ Invalid JSON response: {result}")
                print(f"  ❌ JSON error: {json_error}")
                raise Exception(f"Invalid JSON response: {json_error}")
            
            # Convert to our page type enum
            page_type_map = {
                'SEARCH_PAGE': PageType.SEARCH_PAGE,
                'RESULTS_PAGE': PageType.RESULTS_PAGE,
                'JOB_DETAIL_PAGE': PageType.JOB_DETAIL_PAGE,
                'APPLICATION_PAGE': PageType.APPLICATION_PAGE,
                'LOGIN_PAGE': PageType.LOGIN_PAGE,
                'CAPTCHA_PAGE': PageType.CAPTCHA_PAGE,
                'ERROR_PAGE': PageType.ERROR_PAGE
            }
            
            return {
                'page_type': page_type_map.get(analysis['page_type'], PageType.UNKNOWN_PAGE),
                'confidence': analysis['confidence'],
                'method': 'image_recognition',
                'page_indicators': analysis.get('key_elements', []),
                'reasoning': analysis.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"❌ Error in image detection: {e}")
            return {
                'page_type': PageType.UNKNOWN_PAGE,
                'confidence': 0.0,
                'method': 'image_recognition',
                'page_indicators': []
            }
    
    def _detect_with_gpt(self, page_url: str, title: str) -> Dict[str, Any]:
        """Use GPT-4 to analyze page type based on URL and title"""
        try:
            import openai
            import json as json_module
            
            # Get some page content for context
            body_element = self.page.query_selector('body')
            if body_element:
                body_text = body_element.inner_text()
                if body_text:
                    body_text = body_text[:1000]
                else:
                    body_text = ""
            else:
                body_text = ""
            
            prompt = f"""
            Analyze this webpage and determine what type of page it is.
            
            URL: {page_url}
            Title: {title}
            Content Preview: {body_text[:500]}...
            
            Determine the page type from these options:
            1. SEARCH_PAGE - Job search form page where users enter keywords/location
            2. RESULTS_PAGE - Job search results showing multiple job listings
            3. JOB_DETAIL_PAGE - Individual job posting with full description
            4. APPLICATION_PAGE - Job application form to apply for a position
            5. LOGIN_PAGE - Login/authentication page
            6. CAPTCHA_PAGE - CAPTCHA verification page
            7. ERROR_PAGE - Error page (404, 500, etc.)
            
            Return JSON:
            {{
                "page_type": "SEARCH_PAGE|RESULTS_PAGE|JOB_DETAIL_PAGE|APPLICATION_PAGE|LOGIN_PAGE|CAPTCHA_PAGE|ERROR_PAGE",
                "confidence": 0.0-1.0,
                "reasoning": "Why you think this is the page type",
                "key_indicators": ["list", "of", "indicators"]
            }}
            """
            
            client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing webpages and determining page types for job search automation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate JSON response
            if not result:
                print("  ❌ Empty response from GPT-4")
                raise Exception("Empty response from GPT-4")
            
            # Clean up markdown code blocks if present
            if result.startswith('```json'):
                result = result[7:]  # Remove ```json
            if result.startswith('```'):
                result = result[3:]   # Remove ```
            if result.endswith('```'):
                result = result[:-3]  # Remove trailing ```
            
            result = result.strip()
            
            try:
                analysis = json_module.loads(result)
            except json_module.JSONDecodeError as json_error:
                print(f"  ❌ Invalid JSON response: {result}")
                print(f"  ❌ JSON error: {json_error}")
                raise Exception(f"Invalid JSON response: {json_error}")
            
            # Convert to our page type enum
            page_type_map = {
                'SEARCH_PAGE': PageType.SEARCH_PAGE,
                'RESULTS_PAGE': PageType.RESULTS_PAGE,
                'JOB_DETAIL_PAGE': PageType.JOB_DETAIL_PAGE,
                'APPLICATION_PAGE': PageType.APPLICATION_PAGE,
                'LOGIN_PAGE': PageType.LOGIN_PAGE,
                'CAPTCHA_PAGE': PageType.CAPTCHA_PAGE,
                'ERROR_PAGE': PageType.ERROR_PAGE
            }
            
            return {
                'page_type': page_type_map.get(analysis['page_type'], PageType.UNKNOWN_PAGE),
                'confidence': analysis['confidence'],
                'method': 'gpt_analysis',
                'page_indicators': analysis.get('key_indicators', []),
                'reasoning': analysis.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"❌ Error in GPT detection: {e}")
            return {
                'page_type': PageType.UNKNOWN_PAGE,
                'confidence': 0.0,
                'method': 'gpt_analysis',
                'page_indicators': []
            }
    
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
        return (current_detection['page_type'] == PageType.UNKNOWN_PAGE and 
                current_detection['confidence'] < 0.3)
    
 