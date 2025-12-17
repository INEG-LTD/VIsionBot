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

# Global state for spinner
_spinner_active = False
_spinner_thread = None

def _show_spinner():
    """Show a loading spinner animation"""
    global _spinner_active
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    while _spinner_active:
        sys.stdout.write(f'\r   {spinner_chars[i % len(spinner_chars)]} Thinking...')
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
    """Format action string in first person format: 'I will now [ACTION] the >target<'"""
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
    
    # Format: "I will now [ACTION_TYPE] the >target<"
    return f"I will now [{action_type}] the >{target}<"

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
            print(f"💭 Here's what the agent is thinking: {first_person_reasoning}")
        # Format action in first person
        first_person_action = _format_action_first_person(action)
        print(f"⚡ {first_person_action}")
    
    # Show task completion reasoning only when task is actually complete
    elif event.event_type == EventType.COMPLETION_SUCCESS:
        reasoning = event.details.get('reasoning', '')
        confidence = event.details.get('confidence')
        if reasoning:
            print("\n✅ Task Complete!")
            print(f"📝 Reasoning: {reasoning}")
            if confidence is not None:
                print(f"🎯 Confidence: {confidence:.2f}")
    
    # Also show completion from agent_complete event (backup)
    elif event.event_type == EventType.AGENT_COMPLETE and event.details.get('success', False):
        reasoning = event.details.get('reasoning', '')
        confidence = event.details.get('confidence')
        if reasoning:
            print("\n✅ Task Complete!")
            print(f"📝 Reasoning: {reasoning}")
            if confidence is not None:
                print(f"🎯 Confidence: {confidence:.2f}")

# Ensure the user data directory exists
user_data_path = Path.home() / "Desktop" / "bot_user_data_dir"
user_data_path.mkdir(parents=True, exist_ok=True)

# Create configuration using the new BotConfig API
config = BotConfig(
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        reasoning_level=ReasoningLevel.HIGH
    ),
    execution=ExecutionConfig(
        max_attempts=30
    ),
    elements=ElementConfig(
        overlay_mode="all",
        include_textless_overlays=True,
        selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        overlay_only_planning=True
    ),
    recording=RecordingConfig(
        save_gif=True
    ),
    logging=DebugConfig(
        debug_mode=False  # Set to False to use callbacks only (no debug prints)
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

# Register custom callback to show only essential information
bot.event_logger.register_callback(simple_event_callback)
bot.use(ErrorHandlingMiddleware())
bot.start()
bot.page.goto("https://careers.justeattakeaway.com/global/en/job/R_048650/Senior-Procurement-Specialist-IT-Corporate-Services")
sleep(3)

# Run agentic mode - now returns AgentResult with extracted data
result = bot.execute_task(
    "your job is to apply to the job in this url and you'll be done when the form has been submitted",
    max_iterations=100,
    base_knowledge=[
        """"use this details: 
        first name: John, 
        last name: Doe, 
        email: john.doe@example.com, 
        phone: 07385986448, 
        address line 1: 3 John Street,
        city: Liverpool,
        state: England,
        country: United Kingdom,
        gender: Male""",
        "fill all the required fields in the form",
        "you are allowed to use the best value for the field if the user doesn't provide one",
        "you must always use the upload resume button if it is available to upload the resume",
        "only press Enter after typing in a search field if there are NO visible suggestions or dropdown options to click",
        "if asked to search use the best search box contextually available",
        "if there is a cookie banner, accept all cookies",
    ],
    show_completion_reasoning_every_iteration=False  # Only show when actually complete
)

# Check if task succeeded
if result.success:
    print(f"\n✅ Task completed! Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    # Access extracted data if any
    if result.extracted_data:
        print("\n📊 Extracted Data:")
        for prompt, data in result.extracted_data.items():
            print(f"  {prompt}: {data}")
else:
    print(f"\n❌ Task failed: {result.reasoning}")

input("Press Enter to continue...")
