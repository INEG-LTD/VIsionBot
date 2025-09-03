"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import hashlib
import os
import uuid
import re
from typing import Optional

from playwright.sync_api import Browser, Page, Playwright

from ai_utils import generate_model
from models import VisionPlan, PageElements
from element_detection import OverlayManager, ElementDetector
from action_executor import ActionExecutor
from utils import PageUtils
from vision_utils import draw_bounding_boxes
from goals import GoalMonitor, ClickGoal, GoalStatus, FormFillGoal, NavigationGoal


class BrowserVisionBot:
    """Modular vision-based web automation bot"""

    def __init__(self, page: Page=None, model_name: str="gemini-2.0-flash-lite", max_attempts: int = 10):
        self.page = page
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.started = False
        
    def init_browser(self) -> tuple[Playwright, Browser, Page]:
        p = sync_playwright().start()
        browser = p.chromium.launch_persistent_context(
                    viewport={"width": 1280, "height": 800},
                    user_data_dir=os.path.expanduser(f"~/Library/Application Support/Google/Chrome/Automation_{str(uuid.uuid4())[:8]}"),
                    headless=False,
                    args=[
                        "--no-sandbox", 
                        "--disable-dev-shm-usage", 
                        "--disable-blink-features=AutomationControlled",
                        '--disable-features=VizDisplayCompositor',
                        f"--window-position={0},{0}",
                        f"--window-size={1280},{800}",
                        ],
                    channel="chrome"
                )
        
                    
        pages = browser.pages
        if pages:
            page = pages[0]
        else:
            page = browser.new_page()
        
        return p, browser, page
    
    def start(self) -> None:
        """Start the bot"""
        playwright, browser, page = self.init_browser()
        self.playwright = playwright
        self.browser = browser
        self.page = page
        
        # State tracking
        self.current_attempt = 0
        self.last_screenshot_hash = None
        
        # Initialize components
        self.goal_monitor = GoalMonitor(page)
        self.overlay_manager = OverlayManager(page)
        self.element_detector = ElementDetector(model_name=self.model_name)
        self.action_executor = ActionExecutor(page, self.goal_monitor)
        self.page_utils = PageUtils(page)
        
        self.started = True

    def act(self, goal_description: str, additional_context: str = "", smart_goals: bool = True) -> bool:
        """Main method to achieve a goal using vision-based automation"""
        if not self.started:
            print("❌ Bot not started")
            return False
        
        if self.page.url.startswith("about:blank"):
            print("❌ Page is on the initial blank page")
            return False
        
        print(f"🎯 Starting goal: {goal_description}")
        
        # Clear any existing goals before starting a new goal
        if self.goal_monitor.active_goals:
            print(f"🧹 Clearing {len(self.goal_monitor.active_goals)} previous goals")
            self.goal_monitor.clear_all_goals()
        
        # Reset goal monitor state for fresh start
        self.goal_monitor.reset_retry_requests()
        
        self.goal_monitor.set_user_prompt(goal_description)
        
        # Set up smart goal monitoring if enabled
        if smart_goals:
            self._setup_smart_goals(goal_description, additional_context)
        
        print(f"🔍 Smart goals setup: {self.goal_monitor.active_goals}\n")
        
        for attempt in range(self.max_attempts):
            self.current_attempt = attempt + 1
            print(f"\n--- Attempt {self.current_attempt}/{self.max_attempts} ---")
            
            # Show retry context at the start of each new attempt (but don't reset yet)
            if attempt > 0:  # Don't reset on first attempt
                retry_goals = self.goal_monitor.check_for_retry_requests()
                if retry_goals:
                    print(f"🔄 Starting attempt {self.current_attempt} with retry state from previous attempt")
                    for goal in retry_goals:
                        print(f"   🔄 {goal}: Retry attempt {goal.retry_count}/{goal.max_retries}")
                    # Don't reset retry state here - let it persist until after plan generation
            
            # Check if goal is already achieved
            if smart_goals:
                goal_results = self.goal_monitor.evaluate_goals()
                if any(result.status == GoalStatus.ACHIEVED for result in goal_results.values()):
                    print("✅ Smart goal achieved!")
                    self._print_goal_summary()
                    return True
            
            # Get current page state
            page_info = self.page_utils.get_page_info()
            screenshot = self.page.screenshot(full_page=False)
            
            # Skip if we've seen this exact screenshot recently (avoid loops)
            # BUT allow retry attempts to proceed even with same screenshot
            screenshot_hash = hashlib.md5(screenshot).hexdigest()
            retry_goals = self.goal_monitor.check_for_retry_requests()
            
            if screenshot_hash == self.last_screenshot_hash and not retry_goals:
                print("⚠️ Same screenshot as last attempt, scrolling to break loop")
                self.page_utils.scroll_page()
                continue
            elif screenshot_hash == self.last_screenshot_hash and retry_goals:
                print("🔄 Same screenshot but retry requested - proceeding with retry attempt")
            
            self.last_screenshot_hash = screenshot_hash
            
            # Generate plan using vision model
            plan = self._generate_plan(goal_description, additional_context, screenshot, page_info)
            if not plan or not plan.action_steps:
                print("❌ No valid plan generated")
                continue
            
            # Reset retry state after plan generation (retry context has been used)
            retry_goals = self.goal_monitor.check_for_retry_requests()
            if retry_goals:
                print("🔄 Retry context used in plan generation, resetting retry state")
                self.goal_monitor.reset_retry_requests()
            
            print(f"📋 Generated plan with {len(plan.action_steps)} steps")
            print(f"🤔 Reasoning: {plan.reasoning}")
            
            # Execute the plan
            success = self.action_executor.execute_plan(plan, page_info)
            if success:
                if smart_goals:
                    # Check if goals were achieved during execution
                    goal_results = self.goal_monitor.evaluate_goals()
                    
                    # Check for retry requests after goal evaluation
                    retry_goals = self.goal_monitor.check_for_retry_requests()
                    if retry_goals:
                        print("🔄 Goals requested retry after plan execution - regenerating plan")
                        for goal in retry_goals:
                            print(f"   🔄 {goal}: Retry requested (attempt {goal.retry_count}/{goal.max_retries})")
                        # Don't reset retry requests here - let them persist for the next iteration
                        continue
                    
                    if any(result.status == GoalStatus.ACHIEVED for result in goal_results.values()):
                        print("✅ Smart goal achieved during plan execution!")
                        self._print_goal_summary()
                        return True
                    
                    # Wait a bit for page to update, then check goal again
                    # time.sleep(1)
                    # goal_results = self.goal_monitor.evaluate_goals()
                    
                    # # Check for retry requests after second goal evaluation
                    # retry_goals = self.goal_monitor.check_for_retry_requests()
                    # if retry_goals:
                    #     print("🔄 Goals requested retry after post-execution evaluation - regenerating plan")
                    #     for goal in retry_goals:
                    #         print(f"   🔄 {goal}: Retry requested (attempt {goal.retry_count}/{goal.max_retries})")
                    #     self.goal_monitor.reset_retry_requests()
                    #     continue
                    
                    # if any(result.status == GoalStatus.ACHIEVED for result in goal_results.values()):
                    #     print("✅ Smart goal achieved after executing plan!")
                    #     self._print_goal_summary()
                    #     return True
                
                # If plan executed successfully but no goals achieved, scroll down one viewport height
                # to explore more of the page instead of waiting for duplicate screenshots
                print("📜 Plan executed successfully, scrolling down to explore more content")
                self.page_utils.scroll_page()
                continue
            else:
                # Plan execution failed - check if it was due to retry request
                retry_goals = self.goal_monitor.check_for_retry_requests()
                if retry_goals:
                    print("🔄 Plan execution aborted due to retry request - regenerating plan")
                    # Don't reset retry requests here - let them persist for the next iteration
                    # The retry state will be used to inform the next plan generation
                    continue
                else:
                    print("❌ Plan execution failed for unknown reason")
                    continue
        
        print(f"❌ Failed to achieve goal after {self.max_attempts} attempts")
        if smart_goals:
            self._print_goal_summary()
        return False

    def goto(self, url: str) -> None:
        """Go to a URL"""
        if not self.started:
            print("❌ Bot not started")
            return
        
        self.page.goto(url, wait_until="domcontentloaded")
        self.url = url
        
    def _generate_plan(self, goal_description: str, additional_context: str, screenshot: bytes, page_info) -> Optional[VisionPlan]:
        """Generate an action plan using numbered element overlays"""
        
        # Step 1: Create numbered overlays and get element data
        element_data = self.overlay_manager.create_numbered_overlays(page_info)
        
        if not element_data:
            print("❌ No interactive elements found for overlays")
            return None
        
        # Step 2: Take screenshot with numbered overlays visible
        screenshot_with_overlays = self.page.screenshot(full_page=False)
        
        # Step 3: Use AI to identify relevant elements
        detected_elements = self.element_detector.detect_elements_with_overlays(
            goal_description, additional_context, screenshot_with_overlays, element_data, page_info
        )
        
        # Step 4: Clean up overlays after detection
        self.overlay_manager.remove_overlays()
        
        # Step 5: Save debugging image
        if detected_elements and detected_elements.elements:
            bounding_box_image = draw_bounding_boxes(screenshot, detected_elements.elements)
            with open("bounding_box_image.png", "wb") as f:
                f.write(bounding_box_image)
        
        if not detected_elements:
            print("❌ No elements detected, cannot create plan")
            return None
        
        # Step 6: Create action plan using detected elements
        plan = self._create_plan(goal_description, additional_context, detected_elements, page_info)
        
        return plan

    def _create_plan(self, goal_description: str, additional_context: str, detected_elements: PageElements, page_info) -> Optional[VisionPlan]:
        """Create an action plan using AI"""
        print(f"Creating plan for goal: {goal_description}\n")
        
        # Get goal descriptions from active goals to provide more context
        # These descriptions contain detailed, real-time analysis of what each goal needs
        # and help align the AI plan generation with actual goal requirements
        goal_descriptions = []
        if self.goal_monitor.active_goals:
            print("📋 Gathering goal descriptions for plan generation...")
            for goal in self.goal_monitor.active_goals:
                try:
                    # Create a basic context for goal description
                    from goals.base import GoalContext, BrowserState
                    basic_context = GoalContext(
                        initial_state=BrowserState(
                            timestamp=0, url=page_info.url, title=page_info.title,
                            page_width=page_info.width, page_height=page_info.height,
                            scroll_x=0, scroll_y=0
                        ),
                        current_state=BrowserState(
                            timestamp=0, url=page_info.url, title=page_info.title,
                            page_width=page_info.width, page_height=page_info.height,
                            scroll_x=0, scroll_y=0
                        ),
                        page_reference=self.page
                    )
                    
                    goal_desc = goal.get_description(basic_context)
                    goal_descriptions.append(goal_desc)
                    print(f"   📝 {goal.__class__.__name__}: {goal_desc[:100]}...")
                except Exception as e:
                    print(f"   ⚠️ Error getting description for {goal.__class__.__name__}: {e}")
                    goal_descriptions.append(f"{goal.__class__.__name__}: {goal.description}")
        
        # Check for retry context
        retry_goals = self.goal_monitor.check_for_retry_requests()
        retry_context = ""
        if retry_goals:
            retry_info = []
            for goal in retry_goals:
                retry_info.append(f"- {goal.__class__.__name__}: Retry attempt {goal.retry_count}/{goal.max_retries}")
            retry_context = f"""
        
        RETRY CONTEXT:
        The following goals have requested retries due to previous failures:
        {chr(10).join(retry_info)}
        
        IMPORTANT: This is a retry attempt. The previous plan failed because the goals detected issues.
        Make sure to:
        - Look for different elements that might match the goal requirements
        - Consider alternative approaches to achieve the same goal
        - Be more careful about element selection and targeting
        """
            print(f"🔄 Retry context included in plan generation: {len(retry_goals)} goals requesting retry")
        
        # Build the system prompt with goal descriptions
        # Goal descriptions provide structured, real-time analysis of goal requirements
        # They include status indicators (✅ ⏳ ❌ 🎯 📊) and specific actionable guidance
        goal_context = ""
        if goal_descriptions:
            goal_context = f"""
        USER GOAL: {goal_description}
        
        ACTIVE GOAL DESCRIPTIONS:
        {chr(10).join(f"- {desc}" for desc in goal_descriptions)}
        
        IMPORTANT: Use the goal descriptions above to understand exactly what needs to be accomplished. 
        These descriptions contain real-time analysis of the current page state and goal progress.
        Pay special attention to:
        - Which specific fields need to be filled (don't fill unnecessary fields)
        - Current completion status of fields (✅ = completed, ⏳ = needs filling)
        - Any validation errors that need to be addressed (❌ indicators)
        - Specific targets for click or navigation goals
        - Progress metrics (📊 indicators show completion ratios)
        {retry_context}
        """
        
        system_prompt = f"""
        You are a web automation assistant. Create a plan based on the ACTIVE GOAL DESCRIPTIONS below.
        
        Current page: {page_info.url}
        DETECTED ELEMENTS: {[elem.model_dump() for elem in detected_elements.elements]}
        {goal_context}
        
        CRITICAL INSTRUCTIONS:
        1. Focus ONLY on the ACTIVE GOAL DESCRIPTIONS and USER GOAL above
        2. The goal descriptions contain real-time analysis of what actually needs to be done
        3. Follow the specific requirements in the goal descriptions exactly:
           - Only fill fields marked as "NEEDS FILLING" or "EMPTY"
           - Ignore fields marked as "COMPLETED" or "✅"
           - Address any validation errors mentioned
           - Target specific elements described in click/navigation goals
        4. Create a plan with 1-3 action steps based on the goal descriptions
        5. USE SPECIALIZED ACTION TYPES for special fields:
           - HANDLE_SELECT: For select dropdowns
           - HANDLE_UPLOAD: For file upload fields
           - HANDLE_DATETIME: For date/time input fields
           - Use CLICK/TYPE only for simple interactions.
        
        Return a VisionPlan with action_steps, reasoning, and confidence.
        """
        
        # Debug: Print retry context if present
        if retry_goals:
            print(f"🔍 System prompt includes retry context: {retry_context[:200]}...")
        
        try:
            plan = generate_model(
                prompt="Create a plan to achieve the goal based on the ACTIVE GOAL DESCRIPTIONS.",
                model_object_type=VisionPlan,
                reasoning_level="medium",
                system_prompt=system_prompt,
                model="gpt-5-nano"
            )
            return plan
        except Exception as e:
            print(f"❌ Error creating plan: {e}")
            return None

    def _setup_smart_goals(self, goal_description: str, additional_context: str = "") -> None:
        """Set up smart goals based on the goal description"""
        goal_lower = goal_description.lower().strip()
        
        # Detect form filling goals
        form_patterns = [
            "fill", "complete", "submit", "form", "enter", "input", "set the", "set only"
        ]
        
        if any(pattern in goal_lower for pattern in form_patterns):
            form_goal = FormFillGoal(
                description=f"Form fill goal: {goal_description}",
                trigger_on_submit=False,
                trigger_on_field_input=True
            )
            self.goal_monitor.add_goal(form_goal)
            print(f"📝 Added FormFillGoal: '{goal_description}'")
        
        # Detect click goals
        click_patterns = ["click", "tap", "press", "select", "choose", "close"]
        if any(pattern in goal_lower for pattern in click_patterns):
            target_description = self._extract_click_target(goal_description)
            if target_description:
                click_goal = ClickGoal(
                    description=f"Click goal: {goal_description}",
                    target_description=target_description
                )
                self.goal_monitor.add_goal(click_goal)
                print(f"🎯 Added ClickGoal: '{target_description}'")
        
        # Detect navigation goals
        navigation_patterns = ["go to", "navigate to", "open", "visit"]
        if any(pattern in goal_lower for pattern in navigation_patterns):
            navigation_intent = self._extract_navigation_intent(goal_description)
            if navigation_intent:
                nav_goal = NavigationGoal(
                    description=f"Navigation goal: {goal_description}",
                    navigation_intent=navigation_intent
                )
                self.goal_monitor.add_goal(nav_goal)
                print(f"🧭 Added NavigationGoal: '{navigation_intent}'")

    def _extract_click_target(self, goal_description: str) -> Optional[str]:
        """Extract what should be clicked from a goal description"""
        patterns = [
            r"click (?:on )?(?:the )?(.+)",
            r"tap (?:on )?(?:the )?(.+)",
            r"press (?:the )?(.+)",
            r"select (?:the )?(.+)",
            r"choose (?:the )?(.+)",
            r"close (?:the )?(.+)"
        ]
        
        goal_lower = goal_description.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_navigation_intent(self, goal_description: str) -> Optional[str]:
        """Extract navigation intent from a goal description"""
        patterns = [
            r"go to (?:the )?(.+)",
            r"navigate to (?:the )?(.+)",
            r"open (?:the )?(.+)",
            r"visit (?:the )?(.+)"
        ]
        
        goal_lower = goal_description.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                intent = match.group(1).strip()
                intent = re.sub(r'\s+(page|section|area)$', r' \1', intent)
                return intent
        
        return None

    def _print_goal_summary(self) -> None:
        """Print a summary of all goal statuses"""
        summary = self.goal_monitor.get_status_summary()
        
        print("\n📊 Goal Summary:")
        print(f"   Total Goals: {summary['total_goals']}")
        print(f"   ✅ Achieved: {summary['achieved']}")
        print(f"   ⏳ Pending: {summary['pending']}")
        print(f"   ❌ Failed: {summary['failed']}")

    # Convenience methods
    def click_element(self, description: str) -> bool:
        return self.act(f"Click the {description}")

    def fill_form_field(self, field_name: str, value: str) -> bool:
        return self.act(f"Fill the {field_name} field with '{value}'")


# Example usage
if __name__ == "__main__":
    from playwright.sync_api import sync_playwright

    bot = BrowserVisionBot()
    bot.start()
    bot.goto("https://google.com/")
    # try:  
    while True:
        user_input = input('\n👤 New task or "q" to quit: ')
        if user_input.lower() == "q":
            break
        bot.act(user_input)
    # if success:
    #     print("🎉 Goal achieved successfully!")
    # else:
    #     print("😞 Failed to achieve goal")
    # finally:
    #     bot.browser.close()
    #     bot.playwright.stop()
