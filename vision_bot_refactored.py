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
from models.core_models import ActionType
from element_detection import OverlayManager, ElementDetector
from action_executor import ActionExecutor
from models.core_models import PageInfo
from utils import PageUtils
from vision_utils import draw_bounding_boxes
from goals import GoalMonitor, ClickGoal, GoalStatus, FormFillGoal, NavigationGoal, IfGoal, PressGoal, ScrollGoal, Condition, BaseGoal
from goals.condition_utils import create_condition_from_text


class BrowserVisionBot:
    """Modular vision-based web automation bot"""

    def __init__(self, page: Page=None, model_name: str="gemini-2.0-flash", max_attempts: int = 10):
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
        self.page_utils = PageUtils(page)
        self.action_executor = ActionExecutor(page, self.goal_monitor, self.page_utils)
        
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
            goal_description = self._setup_smart_goals(goal_description, additional_context)
        
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
            
            # Generate plan using vision model (conditional goals are already resolved to their sub-goals)
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
            
            # Check if plan contains a STOP action
            if any(step.action == ActionType.STOP for step in plan.action_steps):
                print("🛑 Plan contains STOP action - terminating automation")
                return True
            
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
        


    def _generate_plan(self, goal_description: str, additional_context: str, screenshot: bytes, page_info: PageInfo) -> Optional[VisionPlan]:
        """Generate an action plan using numbered element overlays"""
        
        # Check if any active goal needs element detection
        needs_detection = any(goal.needs_detection for goal in self.goal_monitor.active_goals) if self.goal_monitor.active_goals else True
        
        if not needs_detection:
            print("🚫 Skipping element detection - goal doesn't require it")
            # Create empty detected elements for goals that don't need detection
            detected_elements = PageElements(elements=[])
        else:
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
        plan = self._create_plan(goal_description, additional_context, detected_elements, page_info, screenshot)
        
        return plan

    def _create_plan(self, goal_description: str, additional_context: str, detected_elements: PageElements, page_info: PageInfo, screenshot: bytes) -> Optional[VisionPlan]:
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
                            page_width=page_info.doc_width, page_height=page_info.doc_height,
                            scroll_x=0, scroll_y=0
                        ),
                        current_state=BrowserState(
                            timestamp=0, url=page_info.url, title=page_info.title,
                            page_width=page_info.doc_width, page_height=page_info.doc_height,
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
                if goal.retry_reason:
                    retry_info.append(f"  Reason: {goal.retry_reason}")
            retry_context = f"""
        
        RETRY CONTEXT:
        The following goals have requested retries due to previous failures:
        {chr(10).join(retry_info)}
        
        IMPORTANT: This is a retry attempt. The previous plan failed because the goals detected issues.
        Make sure to:
        - Look for different elements that might match the goal requirements
        - Consider alternative approaches to achieve the same goal
        - Be more careful about element selection and targeting
        - Address the specific failure reasons mentioned above
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
           - PRESS: For keyboard key presses (e.g., 'enter', 'tab', 'ctrl+c', 'escape')
           - STOP: To terminate automation and return True
           - Use CLICK/TYPE only for simple interactions. Only use when there is no appropriate action to take or you don't know what to do.
        
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
                model="gpt-5-mini",
                image=screenshot
            )
            return plan
        except Exception as e:
            print(f"❌ Error creating plan: {e}")
            return None


    def _setup_smart_goals(self, goal_description: str, additional_context: str = "") -> str:
        """Set up smart goals based on the goal description and return the updated description"""
        goal_lower = goal_description.lower().strip()
        
        # Detect conditional goals first (they can contain other goal types as sub-goals)
        conditional_patterns = [
            "if", "when", "unless", "provided that", "in case", "should", 
            "conditional", "depending on", "based on", "if then", "if else"
        ]
        if any(pattern in goal_lower for pattern in conditional_patterns):
            conditional_goal = self._create_conditional_goal(goal_description)
            if conditional_goal:
                # Evaluate the conditional goal immediately to get the active sub-goal and updated description
                active_sub_goal, updated_description = self._evaluate_conditional_goal_immediately(conditional_goal)
                if active_sub_goal:
                    self.goal_monitor.add_goal(active_sub_goal)
                    print(f"🔀 Evaluated conditional goal and added active sub-goal: {active_sub_goal.__class__.__name__} - '{active_sub_goal.description}'")
                    # Return the updated goal description to focus on the specific action
                    return updated_description
                else:
                    print("⚠️ Conditional goal evaluation failed")
                    return goal_description  # Return original if evaluation failed
        
        # Use shared goal creation logic
        goal = self._create_goal_from_description(goal_description, is_sub_goal=False)
        if goal:
            self.goal_monitor.add_goal(goal)
            print(f"✅ Added {goal.__class__.__name__}: '{goal_description}'")
        
        # Return the original goal description for non-conditional goals
        return goal_description

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

    def _extract_press_target(self, goal_description: str) -> Optional[str]:
        """Extract what keys should be pressed from a goal description"""
        patterns = [
            r"press (?:the )?(enter|tab|escape|space|backspace|delete|home|end|pageup|pagedown|up|down|left|right|f1|f2|f3|f4|f5|f6|f7|f8|f9|f10|f11|f12)",
            r"press (?:the )?(ctrl|alt|shift)\+([a-z0-9])",
            r"press (?:the )?(cmd|command)\+([a-z0-9])",
            r"press (?:the )?(control)\+([a-z0-9])",
            r"press (?:the )?(option)\+([a-z0-9])"
        ]
        
        goal_lower = goal_description.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                if len(match.groups()) == 1:
                    # Single key press
                    return match.group(1).strip()
                elif len(match.groups()) == 2:
                    # Key combination
                    modifier = match.group(1).strip()
                    key = match.group(2).strip()
                    return f"{modifier}+{key}"
        
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

    def _create_conditional_goal(self, goal_description: str) -> Optional[IfGoal]:
        """Create a conditional goal from a goal description"""
        try:
            # Parse the conditional statement
            condition_text, success_action, fail_action = self._parse_conditional_statement(goal_description)
            
            print("🔍 DEBUG: Parsed conditional statement:")
            print(f"   📋 Condition: '{condition_text}'")
            print(f"   ✅ Success action: '{success_action}'")
            print(f"   ❌ Fail action: '{fail_action}'")
            
            if not condition_text:
                print(f"⚠️ Could not parse condition from: {goal_description}")
                return None
            
            # Try to create condition from text using AI parser
            condition = create_condition_from_text(condition_text)
            
            if not condition:
                # Fallback: create a simple condition based on common patterns
                condition = self._create_fallback_condition(condition_text)
            
            if not condition:
                print(f"⚠️ Could not create condition for: {condition_text}")
                return None
            
            print("🔍 DEBUG: Created condition:")
            print(f"   📋 Condition type: {condition.condition_type}")
            print(f"   📋 Condition description: {condition.description}")
            
            # Create sub-goals for success and fail actions
            # A success action is required - if none provided, the goal should fail
            if success_action is None:
                print("⚠️ No success action provided for conditional goal. Goal will fail.")
                return None
            
            success_goal = self._create_sub_goal(success_action)
            fail_goal = self._create_sub_goal(fail_action or "Complete fail action")
            
            print("🔍 DEBUG: Created sub-goals:")
            print(f"   ✅ Success goal: {success_goal.__class__.__name__} - '{success_goal.description}'")
            print(f"   ❌ Fail goal: {fail_goal.__class__.__name__} - '{fail_goal.description}'")
            
            # Create the IfGoal
            if_goal = IfGoal(
                condition=condition,
                success_goal=success_goal,
                fail_goal=fail_goal,
                description=goal_description
            )
            
            return if_goal
            
        except Exception as e:
            print(f"❌ Error creating conditional goal: {e}")
            return None

    def _parse_conditional_statement(self, goal_description: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a conditional statement into condition, success action, and fail action using AI"""
        try:
            from ai_utils import generate_text
            
            # Create a prompt for the AI to parse the conditional statement
            prompt = f"""
                        Parse the following conditional statement and extract the condition, success action, and fail action.

                        Conditional statement: "{goal_description}"

                        Rules:
                        - Extract the condition that needs to be evaluated
                        - Extract the success action to take if the condition is true
                        - Extract the fail action to take if the condition is false (if present)
                        - A success action is REQUIRED - if none is found, the statement is invalid
                        - A fail action is OPTIONAL - if none is found, use null

                        Common conditional patterns:
                        - "if X then Y else Z" → condition: X, success: Y, fail: Z
                        - "when X do Y otherwise Z" → condition: X, success: Y, fail: Z
                        - "if X, Y, else Z" → condition: X, success: Y, fail: Z
                        - "should X, then Y, otherwise Z" → condition: X, success: Y, fail: Z
                        - "provided that X, do Y, else Z" → condition: X, success: Y, fail: Z
                        - "in case X, Y, otherwise Z" → condition: X, success: Y, fail: Z
                        - "unless X, do Y, else Z" → condition: X, success: Y, fail: Z
                        - "if X then Y" → condition: X, success: Y, fail: null
                        - "when X do Y" → condition: X, success: Y, fail: null

                        Response format (JSON only):
                        {{
                            "condition": "the condition to evaluate",
                            "success_action": "action to take if condition is true",
                            "fail_action": "action to take if condition is false (or null if not specified)"
                        }}

                        If the statement cannot be parsed or has no success action, return:
                        {{
                            "condition": null,
                            "success_action": null,
                            "fail_action": null
                        }}
                        """
            
            # Call AI to parse the conditional statement
            response = generate_text(
                prompt=prompt,
                reasoning_level="minimal",
                system_prompt="You are a conditional statement parser. Extract condition, success action, and fail action from natural language conditional statements. Return only JSON.",
                model="gpt-5-nano"
            )
            
            if not response:
                print(f"⚠️ No AI response for conditional statement: '{goal_description}'")
                return None, None, None
            
            # Parse the JSON response
            import json
            try:
                result: dict = json.loads(response.strip())
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON response for conditional statement '{goal_description}': {response}")
                return None, None, None
            
            condition = result.get('condition')
            success_action = result.get('success_action')
            fail_action = result.get('fail_action')
            
            # Validate that we have at least a condition and success action
            if not condition or not success_action:
                print(f"⚠️ Invalid conditional statement - missing condition or success action: '{goal_description}'")
                return None, None, None
            
            # Convert null fail_action to None
            if fail_action is None or fail_action == "null":
                fail_action = None
            
            return condition, success_action, fail_action
                
        except Exception as e:
            print(f"⚠️ Error parsing conditional statement '{goal_description}': {e}")
            return None, None, None

    def _create_fallback_condition(self, condition_text: str) -> Optional[Condition]:
        """Create a fallback condition when AI parser fails"""
        try:
            # Simple fallback conditions based on common patterns
            condition_lower = condition_text.lower().strip()
            
            # Check for cookie banner patterns
            if "cookie" in condition_lower and ("banner" in condition_lower or "consent" in condition_lower):
                # Use common cookie banner selectors
                cookie_selectors = [
                    "[data-testid*='cookie']",
                    ".cookie-banner",
                    "#cookie-banner", 
                    "[class*='cookie']",
                    "[id*='cookie']",
                    "[data-testid*='consent']",
                    ".consent-banner",
                    "[class*='consent']"
                ]
                selector = ", ".join(cookie_selectors)
                from goals.condition_utils import element_exists_condition
                return element_exists_condition(selector, "Check if a cookie banner exists on the page")
            
            # Check for element existence patterns
            if "element" in condition_lower and ("exists" in condition_lower or "visible" in condition_lower):
                # Extract selector if possible
                selector_match = re.search(r"element\s+['\"]([^'\"]+)['\"]", condition_text)
                if selector_match:
                    selector = selector_match.group(1)
                    from goals.condition_utils import element_exists_condition
                    return element_exists_condition(selector, condition_text)
            
            # Check for text content patterns
            if "contains" in condition_lower and ("text" in condition_lower or "page" in condition_lower):
                text_match = re.search(r"['\"]([^'\"]+)['\"]", condition_text)
                if text_match:
                    text = text_match.group(1)
                    from goals.condition_utils import text_contains_condition
                    return text_contains_condition(text, condition_text)
            
            # Check for URL patterns
            if "url" in condition_lower and "contains" in condition_lower:
                url_match = re.search(r"['\"]([^'\"]+)['\"]", condition_text)
                if url_match:
                    url_fragment = url_match.group(1)
                    from goals.condition_utils import url_contains_condition
                    return url_contains_condition(url_fragment, condition_text)
            
            # Check for weekday patterns
            if "weekday" in condition_lower or "weekend" in condition_lower:
                from goals.condition_utils import is_weekday_condition, is_weekend_condition
                if "weekday" in condition_lower:
                    return is_weekday_condition(condition_text)
                else:
                    return is_weekend_condition(condition_text)
            
            # Default: create a simple computational condition
            def simple_evaluator(context):
                # This is a placeholder - in practice, you'd want more sophisticated parsing
                return True
            
            from goals.base import create_computational_condition
            return create_computational_condition(condition_text, simple_evaluator)
            
        except Exception as e:
            print(f"⚠️ Error creating fallback condition: {e}")
            return None

    def _create_goal_from_description(self, description: str, is_sub_goal: bool = False) -> Optional[BaseGoal]:
        """Shared method to create goals from descriptions (used by both _setup_smart_goals and _create_sub_goal)"""
        description_lower = description.lower().strip()
        prefix = "action" if is_sub_goal else "goal"
        
        # Detect press goals first (keyboard key presses)
        press_patterns = ["press enter", "press tab", "press escape", "press ctrl", "press alt", "press shift", "press f1", "press f2", "press f3", "press f4", "press f5", "press f6", "press f7", "press f8", "press f9", "press f10", "press f11", "press f12", "press space", "press backspace", "press delete", "press home", "press end", "press pageup", "press pagedown", "press up", "press down", "press left", "press right"]
        if any(pattern in description_lower for pattern in press_patterns):
            target_keys = self._extract_press_target(description)
            if target_keys:
                press_goal = PressGoal(
                    description=f"Press {prefix}: {description}",
                    target_keys=target_keys
                )
                return press_goal
        
        # Detect scroll goals
        scroll_patterns = [
            # Vertical scroll patterns
            "scroll down", "scroll up", "scroll to", "scroll by", "scroll a bit", "scroll slightly", 
            "scroll a page", "scroll one page", "scroll half", "scroll a third", "scroll a quarter",
            "scroll two thirds", "scroll three quarters", "scroll to top", "scroll to bottom",
            "scroll to middle", "scroll to center", "scroll down a bit", "scroll up a bit",
            "scroll down slightly", "scroll up slightly", "scroll down a page", "scroll up a page",
            # Horizontal scroll patterns
            "scroll left", "scroll right", "scroll left a bit", "scroll right a bit",
            "scroll left slightly", "scroll right slightly", "scroll left a page", "scroll right a page",
            "scroll left half", "scroll right half", "scroll left a third", "scroll right a third",
            "scroll left a quarter", "scroll right a quarter", "scroll left two thirds", "scroll right two thirds",
            "scroll left three quarters", "scroll right three quarters", "scroll to left edge", "scroll to right edge",
            "scroll to left middle", "scroll to right middle", "scroll to left center", "scroll to right center"
        ]
        if any(pattern in description_lower for pattern in scroll_patterns):
            scroll_goal = ScrollGoal(
                description=f"Scroll {prefix}: {description}",
                user_request=description
            )
            return scroll_goal
        
        # Detect click goals (more specific patterns, excluding press)
        click_patterns = ["click", "tap", "select", "choose", "close"]
        if any(pattern in description_lower for pattern in click_patterns):
            target_description = self._extract_click_target(description)
            if target_description:
                click_goal = ClickGoal(
                    description=f"Click {prefix}: {description}",
                    target_description=target_description
                )
                return click_goal
        
        # Detect form filling goals (after click goals to avoid conflicts)
        form_patterns = [
            "fill", "complete", "submit", "form", "enter", "input", "set the", "set only", "type", "search"
        ]
        
        if any(pattern in description_lower for pattern in form_patterns):
            form_goal = FormFillGoal(
                description=f"Form fill {prefix}: {description}",
                trigger_on_submit=False,
                trigger_on_field_input=True
            )
            return form_goal
        
        # Detect navigation goals
        navigation_patterns = ["go to", "navigate to", "open", "visit"]
        if any(pattern in description_lower for pattern in navigation_patterns):
            navigation_intent = self._extract_navigation_intent(description)
            if navigation_intent:
                nav_goal = NavigationGoal(
                    description=f"Navigation {prefix}: {description}",
                    navigation_intent=navigation_intent
                )
                return nav_goal
        
        # If no specific goal type is detected, create a generic action goal
        class ActionGoal(BaseGoal):
            def __init__(self, description: str):
                super().__init__(description)
            
            def evaluate(self, context):
                from goals.base import GoalResult
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=0.8,
                    reasoning=f"Action completed: {self.description}"
                )
            
            def get_description(self, context):
                return f"Action: {self.description}"
        
        return ActionGoal(description)

    def _create_sub_goal(self, action_description: str):
        """Create a sub-goal from an action description using shared goal creation logic"""
        return self._create_goal_from_description(action_description, is_sub_goal=True)

    def _evaluate_conditional_goal_immediately(self, conditional_goal) -> tuple[Optional[BaseGoal], str]:
        """Evaluate a conditional goal immediately and return the active sub-goal and updated description"""
        try:
            # Create a basic context for evaluation
            from goals.base import GoalContext, BrowserState
            page_info = self.page_utils.get_page_info()
            
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
            
            # Evaluate the conditional goal to get the active sub-goal
            conditional_goal.evaluate(basic_context)
            active_sub_goal = getattr(conditional_goal, '_current_sub_goal', None)
            condition_result = getattr(conditional_goal, '_last_condition_result', None)
            
            if active_sub_goal:
                print("🔍 DEBUG: Conditional goal evaluated immediately:")
                print(f"   📋 Condition result: {condition_result}")
                print(f"   🎯 Active sub-goal: {active_sub_goal.__class__.__name__} - '{active_sub_goal.description}'")
                
                # Extract the specific action from the sub-goal description
                # Remove the prefix like "Click action:" or "Form fill action:" to get the clean action
                sub_goal_desc = active_sub_goal.description
                updated_description = sub_goal_desc
                
                print(f"   📝 Updated goal description: '{updated_description}'")
                return active_sub_goal, updated_description
            else:
                print("⚠️ Conditional goal evaluation failed, no active sub-goal")
                return None, ""
                
        except Exception as e:
            print(f"⚠️ Error evaluating conditional goal immediately: {e}")
            return None, ""

    def _get_detected_elements_for_goal(self, goal: BaseGoal, page_info) -> str:
        """Get detected elements for a specific goal by running element detection"""
        try:
            # Check if this goal needs element detection
            if not goal.needs_detection:
                return "No element detection needed for this goal type"
            
            # Take a screenshot for element detection
            screenshot = self.page.screenshot(full_page=False)
            
            # Create numbered overlays and get element data
            element_data = self.overlay_manager.create_numbered_overlays(page_info)
            
            if not element_data:
                return "No interactive elements found"
            
            # Use AI to identify relevant elements for this specific goal
            detected_elements = self.element_detector.detect_elements_with_overlays(
                goal.description, "", screenshot, element_data, page_info
            )
            
            # Clean up overlays after detection
            self.overlay_manager.remove_overlays()
            
            if not detected_elements or not detected_elements.elements:
                return "No elements detected for this goal"
            
            # Format the detected elements
            formatted_elements = []
            for i, element in enumerate(detected_elements.elements):
                element_info = f"Element {i}: {element.description} ({element.element_type})"
                if hasattr(element, 'box_2d') and element.box_2d:
                    element_info += f" at {element.box_2d}"
                formatted_elements.append(element_info)
            
            return "\n".join(formatted_elements)
            
        except Exception as e:
            print(f"⚠️ Error detecting elements for goal: {e}")
            return "Error detecting elements"

    def _format_elements_for_prompt(self, elements) -> str:
        """Format detected elements for AI prompt"""
        if not elements or not hasattr(elements, 'elements'):
            return "No elements detected"
        
        formatted_elements = []
        for i, element in enumerate(elements.elements):
            element_info = f"Element {i}: {element.description} ({element.element_type})"
            if hasattr(element, 'box_2d') and element.box_2d:
                element_info += f" at {element.box_2d}"
            formatted_elements.append(element_info)
        
        return "\n".join(formatted_elements) if formatted_elements else "No elements detected"

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
    
    bot.act("click the accept cookies button")
    bot.act("type 'nuro ai' into the search bar")
    bot.act("press enter")
    
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
