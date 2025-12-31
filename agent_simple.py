"""
Simplified agent execution script that shows only essential information:
- Agent iteration number
- What the agent is thinking (reasoning)
- What action it's about to take
- Task completion reasoning (only when task is actually done)

This is a cleaner version of agent_dummy.py with minimal logging output.
"""
from time import sleep
import sys
import threading
from middlewares.error_handling_middleware import ErrorHandlingMiddleware
from pathlib import Path
from browser_provider import BrowserConfig
from bot_config import BotConfig, ModelConfig, ExecutionConfig, RecordingConfig, ElementConfig, DebugConfig, UserMessagesConfig
from bot_config import ActFunctionConfig
from ai_utils import ReasoningLevel
from vision_bot import BrowserVisionBot
from utils.event_logger import EventType
from agent.mini_goal_manager import MiniGoalTrigger, MiniGoalMode, MiniGoalScriptContext
from utils.select_option_utils import SelectOptionError
from pydantic import BaseModel
from typing import Optional
import random

def type_text_sequentially(page, text: str, delay: int = None):
    """
    Type text sequentially into the currently focused element using playwright's keyboard.type.

    Args:
        page: The playwright page object
        text: The text to type
        delay: Delay between keystrokes in milliseconds (default: random 50-150ms)
    """
    if delay is None:
        delay = random.randint(50, 150)

    # Type text sequentially using keyboard.type with delay
    page.keyboard.type(text, delay=delay)
    print(f"Typed text: {text}")

class DropdownSelection(BaseModel):
    """Structured response for dropdown selection analysis"""
    recommended_option: str
    confidence: float = 1.0  # How confident the agent is in this recommendation
    reasoning: Optional[str] = None  # Why this option was chosen

class IsDropdownVisible(BaseModel):
    is_visible: bool

# Global state for spinner
_spinner_active = False
_spinner_thread = None

# Global state to track if completion message has been shown
_completion_shown = False

def _show_spinner():
    """Show a loading spinner animation"""
    global _spinner_active
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while _spinner_active:
        sys.stdout.write(f'\r{spinner_chars[i % len(spinner_chars)]} Thinking...')
        sys.stdout.flush()
        sleep(0.1)
        i += 1
    # Clear the spinner line
    sys.stdout.write('\r' + ' ' * 20 + '\r')
    sys.stdout.flush()

def _start_spinner():
    """Start the thinking spinner"""
    global _spinner_active, _spinner_thread
    _spinner_active = True
    _spinner_thread = threading.Thread(target=_show_spinner, daemon=True)
    _spinner_thread.start()

def _stop_spinner():
    """Stop the thinking spinner"""
    global _spinner_active
    _spinner_active = False
    if _spinner_thread:
        _spinner_thread.join(timeout=0.2)

def _convert_to_first_person(text: str) -> str:
    """Convert reasoning text to first person"""
    if not text:
        return text
    
    text = text.strip()
    
    # If already in first person, return as is
    first_person_starters = ('i ', 'i\'', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d', 'i see', 'i need', 'i should', 'i will', 'i can')
    if text.lower().startswith(first_person_starters):
        return text
    
    # Common patterns to convert to first person
    # Start with "The" -> "I see the"
    if text.startswith('The '):
        text = 'I see ' + text.lower()
    # Start with "To" -> "I need to"
    elif text.startswith('To '):
        text = 'I need ' + text.lower()
    # Start with other capital letters -> "I [lowercase]"
    elif text and text[0].isupper():
        text = 'I ' + text[0].lower() + text[1:]
    # Already lowercase -> "I [text]"
    else:
        text = 'I ' + text
    
    # Capitalize the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def _format_action_first_person(action: str) -> str:
    """Format action string in first person format: 'I will now [ACTION] >target<'"""
    if not action:
        return action
    
    # Parse action format: "action_type: target"
    # Examples: "click: Apply now button", "type: John in name field", "scroll: down"
    parts = action.split(':', 1)
    if len(parts) != 2:
        # If no colon, return as is with first person prefix
        return f"I will now {action}"
    
    action_type = parts[0].strip().upper()
    target = parts[1].strip()
    
    # Format: "I will now [ACTION_TYPE] >target<"
    return f"I will now [{action_type}] >{target}<"

def setup_mini_goals(bot: BrowserVisionBot):
    """Register various mini goal examples"""

    # Example 1: Autonomous dropdown handling
    # When the agent clicks on any dropdown, it automatically focuses on selecting the right option
    dropdown_trigger_select = MiniGoalTrigger(
        action_type="select",
        target_regex=r"(?i)dropdown|select|combobox"
    )
    dropdown_trigger_click = MiniGoalTrigger(
        action_type="click",
        target_regex=r"(?i)dropdown|select|combobox"
    )
    
    def select_dropdown_handler(context: MiniGoalScriptContext):
        """Handle dropdown selection with intelligent option analysis"""
        print("🎯 Running dropdown selection mini-goal...")

        try:
            # Get the current action that triggered this mini-goal
            current_action = context.action

            if not current_action:
                print("❌ No current action available for dropdown selection")
                return

            print(f"📍 Action to execute: {current_action}")

            # Parse the action to understand what we need to do
            action_part = ""

            if current_action.startswith('select:'):
                # Handle select actions: "select: Python in programming language dropdown"
                action_part = current_action[7:].strip()  # Remove "select:" prefix
                print(f"🎯 Need to select: '{action_part}'")

                # remove any dropdown/combobox/select text from the action_parts
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")
            elif current_action.startswith('click:'):
                # Handle click actions on dropdown elements: "click: 'Select theme...' button"
                # Extract what was clicked and infer the selection context
                action_part = current_action[6:].strip()  # Remove "click:" prefix
                print(f"🎯 Clicked on: '{action_part}'")

                # remove any dropdown/combobox/select text from the click_description
                action_part = action_part.replace("dropdown", "").replace("combobox", "").replace("select", "")

            analysis_prompt = f"""
                Based on the current page state, what is the best option to select for this select field with the placeholder: "{action_part}"?
                Consider the overall task context and what would be the most logical selection.
                """
            selection_info: DropdownSelection = context.ask_question_structured(
                analysis_prompt,
                DropdownSelection
            )

            dropdown_prompt = f"""
                Based on the current page state, is the dropdown with the placeholder: "{action_part}" visible?
                """
            dropdown_visible: IsDropdownVisible = context.ask_question_structured(
                dropdown_prompt,
                IsDropdownVisible
            )

            print(f"🤖 AI Analysis: Select '{selection_info.recommended_option}'")
            print(f"   Confidence: {selection_info.confidence}")

            # Skip selection if confidence is too low
            if selection_info.confidence < 0.3:
                print("⚠️ AI confidence too low, skipping selection")
                return
        
            # if dropdown_visible.is_visible:
            #     bot.act(f"click: {current_action}")
                
            # sleep(1)
            bot.act(f"type: {selection_info.recommended_option} in {action_part}")
            sleep(5)
            bot.act(f"click: {selection_info.recommended_option}")
        except SelectOptionError as e:
            print(f"❌ Select option error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in dropdown handler: {e}")

    bot.register_mini_goal(
        trigger=dropdown_trigger_click,
        mode=MiniGoalMode.SCRIPTED,
        handler=select_dropdown_handler,
        instruction_override="A specialized dropdown selection handler will analyze available options and select the most appropriate one based on context."
    )
    bot.register_mini_goal(
        trigger=dropdown_trigger_select,
        mode=MiniGoalMode.SCRIPTED,
        handler=select_dropdown_handler,
        instruction_override="A specialized dropdown selection handler will analyze available options and select the most appropriate one based on context."
    )
    # Example 3: Observation-based mini goal
    # Trigger when a specific error message appears
    def error_recovery_handler(context: MiniGoalScriptContext):
        """Handle error messages that appear on the page"""
        print("🚨 Error detected, running recovery mini-goal...")

        # Get more context about the error
        error_details = context.ask_question(
            "An error message appeared on the page. What type of error is this and how should I handle it? "
            "Consider whether it's a validation error, network error, or user input error."
        )

        # Try common recovery actions
        if "network" in error_details.lower():
            context.bot.page.reload()
        elif "login" in error_details.lower() or "auth" in error_details.lower():
            context.ask_question("The user needs to log in. Should I navigate to the login page or ask them for credentials?")
        else:
            # For other errors, highlight the error and ask for guidance
            context.bot.page.evaluate("""
                const errors = document.querySelectorAll('.error, .alert-danger, [class*="error"]');
                errors.forEach(el => el.style.backgroundColor = 'yellow');
            """)

    error_trigger = MiniGoalTrigger(
        observation_regex=r"(?i)error|failed|invalid|please try again|something went wrong"
    )

    bot.register_mini_goal(
        trigger=error_trigger,
        mode=MiniGoalMode.SCRIPTED,
        handler=error_recovery_handler
    )

    # Example 4: Complex multi-step workflow
    # Handle file upload workflows
    def file_upload_handler(context: MiniGoalScriptContext):
        """Guide the user through file upload process"""
        print("📁 File upload mini-goal activated...")

        # Check what type of files are expected
        upload_requirements = context.ask_question(
            "What type of file should be uploaded here? Consider file format, size limits, "
            "and any specific naming conventions or content requirements."
        )

        print(f"🤖 Upload requirements: {upload_requirements}")

        # Check current page state
        upload_state = context.bot.page.evaluate("""
            () => {
                const fileInputs = document.querySelectorAll('input[type="file"]');
                const dragZones = document.querySelectorAll('[class*="drop"], [class*="upload"]');
                return {
                    fileInputs: fileInputs.length,
                    dragZones: dragZones.length,
                    hasProgress: !!document.querySelector('[class*="progress"], .upload-progress')
                };
            }
        """)

        if upload_state['fileInputs'] > 0:
            context.ask_question(
                f"I found {upload_state['fileInputs']} file input fields. "
                "Should I ask the user to select a file, or do they want me to use a test file?"
            )
        elif upload_state['dragZones'] > 0:
            context.ask_question(
                "This appears to be a drag-and-drop upload interface. "
                "Should I guide the user through the drag-and-drop process?"
            )

    upload_trigger = MiniGoalTrigger(
        action_type="click",
        target_regex=r"(?i)upload.*file|choose.*file|select.*file|browse"
    )

    bot.register_mini_goal(
        trigger=upload_trigger,
        mode=MiniGoalMode.SCRIPTED,
        handler=file_upload_handler
    )

# Custom callback to show only essential information
def simple_event_callback(event):
    """
    Custom callback that only shows:
    - Agent iteration number
    - What the agent is thinking (reasoning)
    - What action it's about to take
    - Task completion reasoning (only when task is actually done)
    """
    # Show iteration number
    if event.event_type == EventType.AGENT_ITERATION:
        iteration = event.details.get('iteration', '?')
        max_iterations = event.details.get('max_iterations', '?')
        print(f"\n🔄 Iteration {iteration}/{max_iterations}")
        # Start spinner while thinking
        _start_spinner()
    
    # Show what the agent is thinking and what action it's about to take
    elif event.event_type == EventType.ACTION_DETERMINED:
        # Stop spinner
        _stop_spinner()
        
        action = event.details.get('action', 'Unknown action')
        reasoning = event.details.get('reasoning', '')
        
        if reasoning:
            # Convert to first person
            first_person_reasoning = _convert_to_first_person(reasoning)
            print(f"> Here's what the agent is thinking: {first_person_reasoning}")
        # Format action in first person
        first_person_action = _format_action_first_person(action)
        print(f"    ⚡ {first_person_action}")
    
    # Show task completion reasoning only when task is actually complete
    # elif event.event_type == EventType.COMPLETION_SUCCESS:
    #     reasoning = event.details.get('reasoning', '')
    #     confidence = event.details.get('confidence')
    #     if reasoning:
    #         print("\n✅ Task Complete!")
    #         print(f"📝 Reasoning: {reasoning}")
    #         if confidence is not None:
    #             print(f"🎯 Confidence: {confidence:.2f}")
    
    # Also show completion from agent_complete event (backup - only if COMPLETION_SUCCESS didn't fire)
    elif event.event_type == EventType.AGENT_COMPLETE and event.details.get('success', False):
        # Only show if we haven't already shown a completion message
        # This is a fallback in case COMPLETION_SUCCESS event doesn't fire
        reasoning = event.details.get('reasoning', '')
        confidence = event.details.get('confidence')
        if reasoning:
            # Convert reasoning to first person
            first_person_reasoning = _convert_to_first_person(reasoning)
            print("\n✅ The task has been completed!")
            print(f"📝 Reasoning: {first_person_reasoning}")
            if confidence is not None:
                print(f"🎯 Confidence: {confidence:.2f}")

# Ensure the user data directory exists
user_data_path = Path.home() / "Desktop" / "bot_user_data_dir"
user_data_path.mkdir(parents=True, exist_ok=True)

# Create configuration using the new BotConfig API
config = BotConfig(
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="gpt-5-mini",
        # command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        reasoning_level=ReasoningLevel.NONE
    ),
    execution=ExecutionConfig(
        max_attempts=30
    ),
    elements=ElementConfig(
        overlay_mode="all",
        include_textless_overlays=True,
        selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        # overlay_only_planning=True,
        include_overlays_in_agent_context=False,
        # max_coordinate_overlays=100  # Limit overlays for better performance
    ),
    recording=RecordingConfig(
        save_gif=True
    ),
    logging=DebugConfig(
        debug_mode=True  # Set to False to use callbacks only (no debug prints)
    ),
    browser=BrowserConfig(
        provider_type="persistent",
        headless=False,
        apply_stealth=False,
        user_data_dir=str(user_data_path)
    ),
    act_function=ActFunctionConfig(
        enable_target_context_guard=False,
        enable_modifier=True,
        enable_additional_context=True
    ),
    user_messages=UserMessagesConfig(
        file_upload_prompt="⏸️ Please select the file you would like to upload. Press [Enter] when done...",
        file_upload_interrupted="⚠️ Upload interrupted. Please try again."
    )
)

# Create bot with config
bot = BrowserVisionBot(config=config)
setup_mini_goals(bot)

# Register custom callback to show only essential information
bot.event_logger.register_callback(simple_event_callback)
bot.use(ErrorHandlingMiddleware())
bot.start()
bot.page.goto("https://careers.capgemini.com/job/London-IOS-Developer/1264832101/?feedId=388933")

# Wait 15 seconds before starting the task
# sleep(15)

# Run agentic mode - now returns AgentResult with extracted data
result = bot.execute_task(
    "your job is to navigate to the job form page, fill the form with the data provided and submit the form. you'll be done when the form has been submitted",
    base_knowledge=[
        # Form data (structured)
        """Form data to use:
        - First name: John
        - Last name: Doe
        - Email: john.doe@example.com
        - Phone: 07385986448
        - Address: 3 John Street
        - City: Liverpool
        - State: England
        - Country: United Kingdom
        - Gender: Male""",
        
        # Behavioral rules (concise)
        "For search fields: Only press Enter if NO dropdown suggestions are visible",
        "For file uploads: Prefer 'Upload from Device' or 'Local File' over cloud storage options",
        "File picker dialogs are handled automatically - do not interact with them",
        "Use best judgment for missing field values that align with provided data",
        "Don't apply via LinkedIn - use the standard form instead"
    ],
    show_completion_reasoning_every_iteration=False,  # Only show when actually complete
    exploration_mode_enabled=False  # Disable exploration mode for faster execution
)

# Check if task succeeded
# if result.success:
#     print(f"\n✅ Task completed! Confidence: {result.confidence:.2f}")
#     print(f"Reasoning: {result.reasoning}")
    
#     # Access extracted data if any
#     if result.extracted_data:
#         print("\n📊 Extracted Data:")
#         for prompt, data in result.extracted_data.items():
#             print(f"  {prompt}: {data}")
# else:
#     print(f"\n❌ Task failed: {result.reasoning}")

input("Press Enter to continue...")
