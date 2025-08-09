#!/usr/bin/env python3
"""
Test script for the Job Detail to Form Transition Handler

This script demonstrates how the transition handler integrates with the job finding
and form filling processes.
"""

import time
import asyncio
import sys

def test_transition_handler():
    """Test the complete flow: find jobs -> transition to form -> fill form"""
    
    # Check if we're running in an asyncio context
    try:
        asyncio.get_running_loop()
        print("❌ Error: This script cannot run inside an asyncio event loop")
        print("Please run this script directly, not from within an async context")
        print("If you're running this from an IDE or Jupyter notebook, try running it from the terminal:")
        print("   python test_transition_handler.py")
        return False
    except RuntimeError:
        pass  # No asyncio loop running, which is good
    
    # Test preferences
    preferences = {
        'job_titles': ['software engineer', 'developer', 'programmer'],
        'locations': ['remote', 'london', 'new york'],
        'salary_min': 50000,
        'employment_types': ['Full-time'],
        'required_skills': ['python', 'javascript', 'react'],
        'experience_levels': ['Mid', 'Senior'],
        'visa_sponsorship_required': False,
        'remote_flexibility': ['Remote', 'Hybrid'],
        'desired_benefits': ['Health Insurance', 'Stock Options'],
        'exclude_keywords': ['unpaid', 'internship']
    }
    
    try:
        from find_jobs import FindJobsBot
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        return False
    
    # Initialize the bot
    bot = FindJobsBot(headless=False, preferences=preferences)
    
    try:
        print("🧪 Testing Job Detail to Form Transition Handler")
        print("=" * 60)
        
        # Start browser and navigate to a job site
        if not bot.start_browser():
            print("❌ Failed to start browser")
            return False
        
        # Navigate to a job site (you can change this URL)
        job_site_url = "https://www.reed.co.uk/jobs?flow=drawer&registrationSource=job_drawer_search_results&jobId=54227044"
        if not bot.open_url(job_site_url):
            print("❌ Failed to navigate to job site")
            return False
        
        print("✅ Browser started and navigated to job site")
        print("⏸️ The bot will now:")
        print("   1. Find job listings")
        print("   2. Evaluate each job for fit")
        print("   3. Use transition handler to navigate to application forms")
        print("   4. Fill out application forms")
        print("   5. Move to next job listing")
        print("\n🔍 Starting job application process...")
        
        bot.click_accept_cookies_button()
        
        # Instead of calling run_job_application, let's test the components directly
        job_listings = bot.find_job_listings()
        
        if job_listings:
            print(f"✅ Found {len(job_listings)} job listings")
            
            # Test with the first job listing
            if len(job_listings) > 0:
                first_job = job_listings[0]
                print(f"\n📋 Testing with job: '{first_job.title}' at {first_job.company}")
                # Click on the job listing
                try:
                    job_element = bot.page.query_selector(first_job.selector)
                    if job_element:
                        job_element.click()
                        time.sleep(3)
                        print("✅ Clicked on job listing")
                        
                        # Test the transition handler
                        from job_detail_to_form_transition_handler import JobDetailToFormTransitionHandler
                        transition_handler = JobDetailToFormTransitionHandler(bot.page)
                        
                        print("🚀 Testing transition handler...")
                        navigation_result = transition_handler.navigate_to_form()
                        
                        if navigation_result.success:
                            if navigation_result.form_ready:
                                print("✅ Successfully navigated to application form!")
                                print("📝 Form is ready for filling with ApplicationFiller")
                            elif navigation_result.requires_user_intervention:
                                print("⏸️ User intervention required - this is expected for login/CAPTCHA pages")
                            else:
                                print(f"✅ Navigation successful: {navigation_result.action_taken.value}")
                        else:
                            print(f"❌ Navigation failed: {navigation_result.error_message}")
                        
                    else:
                        print("❌ Could not find job element")
                        
                except Exception as e:
                    print(f"❌ Error testing job listing: {e}")
            else:
                print("❌ No job listings found")
        else:
            print("❌ No job listings found")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Keep browser open for inspection
        print("\n⏸️ Browser will remain open for inspection. Close manually when done.")
        try:
            input("Press Enter to close browser...")
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            bot.stop_browser()

if __name__ == "__main__":
    # Ensure we're not in an asyncio context
    try:
        asyncio.get_running_loop()
        print("❌ Error: This script cannot run inside an asyncio event loop")
        print("Please run this script directly, not from within an async context")
        sys.exit(1)
    except RuntimeError:
        pass  # No asyncio loop running, which is good
    
    test_transition_handler() 