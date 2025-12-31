# BrowserVisionBot

A powerful, vision-based web automation framework that uses AI to interact with web pages like a human would. BrowserVisionBot combines computer vision, large language models (LLMs), and Playwright to create intelligent automation agents that can understand and interact with any web interface.

## 🌟 Key Features

- **Vision-Based Automation**: Uses AI vision models to understand web pages visually, not just through DOM inspection
- **Intelligent Agent System**: Autonomous agents that can plan, execute, and adapt to complete tasks
- **Mini Goals System**: Trigger-based sub-objectives that activate automatically when specific conditions are met, allowing agents to handle complex UI interactions like dropdowns with specialized logic
- **Multi-Tab Management**: Sophisticated tab orchestration with sub-agent support for parallel workflows
- **Flexible Action System**: Supports clicks, typing, form filling, file uploads, navigation, and custom actions. Text input fields are automatically cleared before typing to ensure clean input, even when fields contain previous text.
- **Smart Select Handling**: Automatic detection and handling of native `<select>` elements, custom dropdowns, listboxes, and combobox patterns. Agent can see available options in overlays and make intelligent selections. Select elements are prominently marked with `SELECT_FIELD` in element descriptions, and available options are displayed in the `options=` field to help the agent understand when to use `select:` actions instead of `click:`. Conservative detection prevents false positives for regular inputs with lists. When select handler detects a non-select element, it automatically converts the action to a click action using the full bot infrastructure (overlay detection, element finding, etc.), ensuring seamless interaction with suggestion lists and other clickable option elements.
- **Form Field Context Detection**: Automatically detects and includes associated labels/questions for form elements (inputs, radios, checkboxes). This allows the agent to distinguish between similar options (like "Yes" buttons) that belong to different questions, significantly improving accuracy when filling out complex forms.
- **Data Extraction**: Extract structured data from web pages using natural language prompts
- **Stealth Capabilities**: Built-in stealth features to avoid bot detection
- **Middleware System**: Extensible middleware for logging, caching, error handling, and custom behaviors
- **Smart Error Recovery**: Automatic retry logic, fallback strategies, and stuck detection
- **Configuration-Driven**: Type-safe configuration using Pydantic models
- **Optimized Prompts**: Token-efficient prompts reduce LLM costs by ~83% while maintaining accuracy

## ⚡ Performance Optimizations

### Prompt Engineering (v2.0)

The agent system has been optimized for significantly reduced token usage and improved response times:

**Token Reduction:**
- ReactiveGoalDeterminer system prompts: **580 lines → ~100 lines** (83% reduction)
- CompletionContract prompts: **70 lines → ~25 lines** (64% reduction)
- Interaction summaries: Show full context for recent 3 interactions only, concise indicators for older ones (60% reduction)
- Base knowledge: Structured format with task-specific rules removed (73% reduction)

**Key Improvements:**
- Concise command reference tables replace verbose examples (95% token savings)
- Few-shot learning (3-5 examples) instead of exhaustive rule lists
- Eliminated redundancy (rules no longer repeated 3-5 times across prompts)
- Streamlined action format validation and command documentation
- Context-aware summarization (full details for recent, minimal for historical)

**Impact:**
- ~83% reduction in total prompt tokens across all agent operations
- Faster LLM response times due to shorter prompts
- Lower API costs for high-volume automation
- Improved attention to critical rules (no "lost in the middle" effect)
- Maintains same accuracy and reliability with more efficient prompting

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Architecture](#architecture)
- [Examples](#examples)
- [Development](#development)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Playwright
- Google Gemini API key or OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install as a Package

Use editable install during development or install directly from Git:

```bash
# Editable (recommended for development)
pip install -e .

# Or install from Git
pip install git+https://github.com/INEG-LTD/VIsionBot.git
```

### Install Playwright Browsers

```bash
playwright install chromium
```

### Environment Setup

Create a `.env` file in your project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

## 🎯 Quick Start

### Basic Usage

#### Using Context Manager (Recommended)

```python
from browser_vision_bot import BrowserVisionBot, BotConfig, create_browser_provider

# Configure bot
config = BotConfig()
browser_provider = create_browser_provider(config.browser)

# Use context manager for automatic cleanup
with BrowserVisionBot(config=config, browser_provider=browser_provider) as bot:
    # Navigate to a website
    bot.page.goto("https://example.com")
    
    # Perform actions using natural language
    bot.act("Click the 'Get Started' button")
    bot.act("Type 'hello@example.com' into the email field")
    bot.act("Click the submit button")
    
    # Access convenience methods
    print(f"Current URL: {bot.current_url}")
    print(f"Page title: {bot.get_title()}")
    
    # Get session statistics
    stats = bot.session_stats
    print(f"Commands executed: {stats['commands_executed']}")
    
# Bot is automatically cleaned up (end() called) when exiting context
```

#### Manual Management

```python
from playwright.sync_api import sync_playwright
from browser_vision_bot import BrowserVisionBot

# Create a Playwright browser
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    # Initialize the bot
    bot = BrowserVisionBot(page=page)
    bot.start()
    
    # Navigate to a website
    page.goto("https://example.com")
    
    # Perform actions using natural language
    bot.act("Click the 'Get Started' button")
    bot.act("Type 'hello@example.com' into the email field")
    bot.act("Click the submit button")
    
    # Cleanup
    bot.end()
    browser.close()
```

### Using Configuration

```python
from browser_vision_bot import BotConfig, ModelConfig, ExecutionConfig, create_browser_provider

# Create a configuration
config = BotConfig(
    model=ModelConfig(agent_model="gpt-5-mini"),
    execution=ExecutionConfig(max_attempts=15)
)

# Create browser provider
browser_provider = create_browser_provider(config.browser)

# Initialize bot with config
bot = BrowserVisionBot(
    config=config,
    browser_provider=browser_provider
)
bot.start()

# Use the bot
bot.act("Search for 'AI automation' on Google")
```

### Agent Mode (Autonomous)

```python
# Let the agent autonomously complete a task
result = bot.execute_task(
    user_prompt="Find the top 3 Python web frameworks and extract their names and descriptions",
    max_iterations=20
)

if result.success:
    print("Task completed!")
    print("Extracted data:", result.extracted_data)
else:
    print("Task failed:", result.reasoning)
```

## 💡 Core Concepts

### Vision-Based Automation

BrowserVisionBot uses AI vision models to "see" web pages like a human would. Instead of relying solely on DOM selectors, it:

1. Takes screenshots of the page
2. Overlays numbered markers on interactive elements
3. Uses vision AI to identify the correct element based on your description
4. Executes the action on the identified element

This approach is more robust to changes in page structure and works even when traditional selectors fail.

### Agent System

The agent system enables autonomous task completion:

- **Reactive Loop**: Observe → Evaluate Completion → Determine Next Action → Execute → Repeat
- **Completion Evaluation**: LLM-based assessment of whether the task is completed
- **Adaptive Planning**: Dynamically adjusts strategy based on page state and history
- **Stuck Detection**: Identifies when the agent is stuck in a loop and takes corrective action
- **Pause/Resume Control**: Fine-grained pause functionality between actions for debugging and inspection

### Tab Management

Sophisticated multi-tab orchestration:

- **Tab Tracking**: Automatic registration and metadata tracking for all tabs
- **Tab Decisions**: LLM-based decisions on when to switch, close, or create tabs
- **Sub-Agents**: Spawn independent agents in separate tabs for parallel workflows
- **Tab Synchronization**: Keeps all components synchronized when switching tabs

### Pause Functionality

The bot supports pausing agent execution between actions (not just between iterations), providing fine-grained control:

**Why pause between actions?**
- **Granular debugging**: Inspect page state after each individual action completes
- **Immediate feedback**: See results of each action before the agent continues
- **User intervention**: Handle edge cases requiring human judgment
- **State verification**: Verify page state after each action

**How it works:**
1. Pause occurs **between actions**, not mid-action or only at iteration boundaries
2. Thread-safe: Can pause/resume from any thread
3. Two levels of pause:
   - Between agent-determined actions (e.g., between "click: button" and "type: text")
   - Between action steps within a plan (e.g., between click and type within a single plan)

**Example:**
```python
import threading
import time

bot.start()
bot.page.goto("https://example.com")

# Pause after 3 seconds
def pause_after_delay():
    time.sleep(3)
    bot.pause_agent("Manual inspection needed")
    time.sleep(10)  # Keep paused
    bot.resume_agent()

thread = threading.Thread(target=pause_after_delay)
thread.start()

result = bot.execute_task("search for jobs")
```

### Action Execution

Flexible action system with multiple execution strategies:

- **Click**: Mouse clicks with fallback to programmatic clicks
- **Type**: Text input with keyboard simulation
- **Press**: Keyboard key presses (Enter, Tab, Arrow keys, etc.)
- **Scroll**: Scroll to elements or by amount
- **Select**: Dropdown selection
- **Form**: Multi-field form filling
- **Upload**: File upload handling
- **Navigate**: URL navigation with back/forward support

## ⚙️ Configuration

BrowserVisionBot uses Pydantic models for type-safe configuration:

### Configuration Groups

```python
from bot_config import BotConfig, ModelConfig, ExecutionConfig, CacheConfig, RecordingConfig, ErrorHandlingConfig, ActFunctionConfig
from ai_utils import ReasoningLevel

config = BotConfig(
    # AI Model Settings
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="gpt-5-mini",
        reasoning_level=ReasoningLevel.MEDIUM
    ),

    # Execution Behavior
    execution=ExecutionConfig(
        max_attempts=10,
        parallel_completion_and_action=True,
        dedup_mode="auto"
    ),

    # Plan Caching
    cache=CacheConfig(
        enabled=True,
        ttl=6.0,
        max_reuse=1
    ),

    # Recording
    recording=RecordingConfig(
        save_gif=True,
        output_dir="gif_recordings"
    ),

    # Error Handling
    error_handling=ErrorHandlingConfig(
        screenshot_on_error=True,
        max_retries=3,
        retry_delay=2.0
    ),

    # Act Function Parameters
    act_function=ActFunctionConfig(
        enable_target_context_guard=True,
        enable_modifier=True,
        enable_additional_context=True
    )
)
```

### Preset Configurations

```python
# Debug mode - with GIF recording and verbose logging
config = BotConfig.debug()

# Production mode - balanced for reliability
config = BotConfig.production()

# Minimal - all defaults
config = BotConfig.minimal()
```

### Configuration Options

#### ModelConfig
- `model_name`: Default model for all operations (default: "gpt-5-mini")
- `agent_model`: Model used for high-level agent decisions
- `command_model`: Model used for action generation
- `reasoning_level`: Default reasoning level (LOW, MEDIUM, HIGH)

#### ExecutionConfig
- `max_attempts`: Maximum number of attempts for task completion (default: 10)
- `parallel_completion_and_action`: Run completion check and next action in parallel (default: True)
- `dedup_mode`: Deduplication mode: "auto", "on", or "off" (default: "auto")
- `dedup_history_quantity`: Number of interactions to track for dedup (-1 = unlimited)

#### ElementConfig
- `overlay_mode`: Overlay drawing mode (`"interactive"` default, `"all"` includes every visible element)
- `include_textless_overlays`: Keep overlays with no text/aria/placeholder in LLM selection lists, using surrounding DOM context to identify them (default: False)
- `max_detailed_elements`: Maximum number of detailed elements to include (default: 400)
- `max_coordinate_overlays`: Maximum number of coordinate overlays (default: 600)
- `overlay_selection_max_samples`: Limit overlays considered during LLM selection (None for unlimited)

#### CacheConfig
- `enabled`: Enable plan caching (default: True)
- `ttl`: Time-to-live for cached plans in seconds (default: 6.0)
- `max_reuse`: Maximum times a plan can be reused (-1 = unlimited, default: 1)

#### RecordingConfig
- `save_gif`: Enable GIF recording of browser interactions (default: False)
- `output_dir`: Directory for saving GIF recordings (default: "gif_recordings")

#### ErrorHandlingConfig
- `screenshot_on_error`: Take screenshot when errors occur (default: True)
- `screenshot_dir`: Directory for error screenshots (default: "error_screenshots")
- `max_retries`: Maximum retry attempts for recoverable errors (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 2.0)
- `retry_backoff`: Backoff multiplier for exponential retry (default: 2.0)
- `abort_on_critical`: Abort automation on critical errors (default: True)

#### ActFunctionConfig
Control which parameters the agent uses when calling the `act()` function during autonomous execution.

- `enable_target_context_guard`: Enable contextual element filtering (default: True)
- `enable_modifier`: Enable ordinal selection (e.g., "first", "second") (default: True)
- `enable_additional_context`: Enable supplementary information for planning (default: True)

**Example - Disable target_context_guard:**
```python
from bot_config import BotConfig, ActFunctionConfig

config = BotConfig(
    act_function=ActFunctionConfig(
        enable_target_context_guard=False,  # Disable contextual filtering
        enable_modifier=True,
        enable_additional_context=True
    )
)
```

**Use Case:** When the agent is too restrictive in element selection or when you want simpler, more straightforward element targeting without contextual constraints.

**See Also:** Check out `examples/act_function_config_example.py` for more detailed examples and use cases.

### Overlay Customization Example

```python
from bot_config import BotConfig, ElementConfig

config = BotConfig(
    elements=ElementConfig(
        overlay_mode="all",                 # draw overlays on every visible element
        include_textless_overlays=True,     # allow unlabeled overlays in LLM selection (uses surrounding context)
        overlay_selection_max_samples=800,  # widen the candidate list
    )
)
```

## 📚 API Reference

### BrowserVisionBot

Main class for browser automation.

#### Initialization

```python
bot = BrowserVisionBot(
    config: Optional[BotConfig] = None,
    browser_provider: Optional[BrowserProvider] = None,
    page: Page = None,  # Deprecated: use browser_provider
    event_logger: Optional[EventLogger] = None
)
```

#### Context Manager Support

The bot can be used as a context manager for automatic resource cleanup:

```python
# Automatically calls start() and end()
with BrowserVisionBot(config=config) as bot:
    bot.page.goto("https://example.com")
    bot.act("Click the button")
    # end() is automatically called when exiting the context
```

This ensures proper cleanup even if an exception occurs.

#### Core Methods

##### `start() -> None`

Initialize the bot and register the initial page. Must be called before using other methods (unless using context manager).

```python
bot.start()
```

##### `end() -> Optional[str]`

Terminate the bot, stop GIF recording, and prevent any subsequent operations.

```python
gif_path = bot.end()  # Returns path to GIF if recording was enabled
```

**Returns:** `Optional[str]` - Path to generated GIF if recording was enabled, `None` otherwise

##### `act(goal_description, **kwargs) -> ActionResult`

Execute a single action based on natural language description.

```python
result = bot.act("click: login button")
if result.success:
    print(f"Success: {result.message}")
    print(f"Confidence: {result.confidence}")
    print(f"Action ID: {result.metadata.get('action_id')}")
else:
    print(f"Failed: {result.message}")
    print(f"Error: {result.error}")

# Other keyword goals:
# bot.act("type: username in username field")
# bot.act("scroll: down")
# bot.act("press: Enter")
```

**Parameters:**
- `goal_description` (str): Goal description in keyword format (required, cannot be empty). Must use keyword format: "click: button name", "type: text in field", "scroll: down", etc.
- `additional_context` (str, optional): Extra context for the action
- `target_context_guard` (str, optional): Guard condition for actions
- `skip_post_guard_refinement` (bool, optional): Skip refinement after guard checks (default: True)
- `confirm_before_interaction` (bool, optional): Require user confirmation before each action (default: False)
- `action_id` (str, optional): Optional action ID for tracking (auto-generated if not provided)
- `modifier` (List[str], optional): Optional list of modifier strings (deprecated, no longer used)
- `max_attempts` (int, optional): Override bot's max_attempts for this action (must be >= 1)
- `max_retries` (int, optional): Maximum retries for this action (deprecated, no longer used)

**Returns:** `ActionResult` - Structured result with success, message, confidence, and metadata

**Raises:**
- `BotTerminatedError`: If bot has been terminated
- `BotNotStartedError`: If bot is not started
- `ValidationError`: If `goal_description` is empty or `max_attempts` < 1

##### `extract(prompt, **kwargs) -> ActionResult`

Extract data from the current page based on natural language description.

```python
result = bot.extract("Get page title", output_format="text")
if result.success:
    title = result.data  # The extracted text
    print(f"Confidence: {result.confidence}")
    print(f"Message: {result.message}")

# Extract structured data with Pydantic model
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    description: str

result = bot.extract(
    "Extract product information",
    output_format="structured",
    model_schema=Product
)
if result.success:
    product = result.data  # The extracted Product instance
```

**Parameters:**
- `prompt` (str): Natural language description of what to extract (required, cannot be empty)
- `output_format` (str, optional): Format of output - "text", "json", or "structured" (default: "json")
- `model_schema` (Type[BaseModel], optional): Pydantic model for structured output (required if output_format="structured")
- `scope` (str, optional): Extraction scope - "viewport", "full_page", or "element" (default: "viewport")
- `element_description` (str, optional): Required if scope="element"
- `max_retries` (int, optional): Maximum retry attempts if extraction fails (default: 2, must be >= 0)
- `confidence_threshold` (float, optional): Minimum confidence to return result (default: 0.6, must be 0.0-1.0)

**Returns:** `ActionResult` - Structured result with extracted data in `.data` field, plus success, message, confidence, and metadata

**Raises:**
- `BotTerminatedError`: If bot has been terminated
- `BotNotStartedError`: If bot is not started
- `ValidationError`: If `prompt` is empty, invalid `output_format`/`scope`, or missing `element_description`

##### `pause_agent(message: str = "Paused") -> None`

Pause the currently running agent between actions.

When paused, the agent will wait before executing the next action, allowing for manual inspection, debugging, and user intervention. The pause occurs **between actions** (not between iterations), providing fine-grained control.

**Why pause between actions?**
- **Granular control**: Inspect page state after each individual action completes
- **Better debugging**: See immediate results of each action before the agent continues
- **User intervention**: Handle edge cases that require human judgment
- **State verification**: Verify page state after each action completes

**Thread-safe**: Can be called from any thread while the agent is running.

```python
import threading
import time

bot.start()
bot.page.goto("https://example.com")

# Pause after 3 seconds in a background thread
def pause_after_delay():
    time.sleep(3)
    bot.pause_agent("Manual inspection needed")
    time.sleep(10)  # Keep paused for 10 seconds
    bot.resume_agent()

thread = threading.Thread(target=pause_after_delay)
thread.start()

result = bot.execute_task("search for jobs")
```

**Parameters:**
- `message` (str): Optional message to display when paused (default: "Paused")

**Raises:**
- `RuntimeError`: If no agent is currently running

##### `resume_agent() -> None`

Resume the paused agent execution.

Unblocks the agent to continue executing actions. If the agent is not paused, this method has no effect.

**Thread-safe**: Can be called from any thread.

```python
bot.pause_agent("Checking results")
# ... inspect page state ...
bot.resume_agent()  # Continue execution
```

**Raises:**
- `RuntimeError`: If no agent is currently running

##### `is_agent_paused() -> bool`

Check if the agent is currently paused.

```python
if bot.is_agent_paused():
    print("Agent is paused, waiting for resume...")
    bot.resume_agent()
```

**Returns:** `True` if the agent is paused, `False` otherwise

**Raises:**
- `RuntimeError`: If no agent is currently running

##### `execute_task(user_prompt, **kwargs) -> AgentResult`

Execute a task autonomously using the agent.

```python
result = bot.execute_task(
    user_prompt="Research the top 3 AI companies and extract their names",
    max_iterations=20,
    base_knowledge=["Focus on companies founded after 2015"],
    strict_mode=False
)

if result.success:
    print("Extracted data:", result.extracted_data)
    print("Sub-agent results:", result.sub_agent_results)
```

**Parameters:**
- `user_prompt` (str): High-level task description (required, cannot be empty)
- `max_iterations` (int): Maximum agent iterations (default: 50, must be >= 1)
- `track_ineffective_actions` (bool): Track and avoid repeating ineffective actions (default: True)
- `base_knowledge` (List[str], optional): Domain knowledge or constraints that guide agent behavior
- `allow_partial_completion` (bool): Allow completion when major deliverables are satisfied (default: False)
- `check_ineffective_actions` (bool, optional): Override for ineffective-action detection
- `show_completion_reasoning_every_iteration` (bool): Show completion reasoning on every iteration (default: False)
- `strict_mode` (bool): Follow instructions exactly without inferring extra requirements (default: False)
- `clarification_callback` (Callable[[str], str], optional): Callback for asking user clarification questions
- `max_clarification_rounds` (int): Maximum number of clarification rounds (default: 3, must be >= 0)

**Returns:** `AgentResult` object with:
- `success`: Whether the task completed successfully
- `extracted_data`: Dictionary of extracted data (key: extraction prompt, value: extracted result)
- `reasoning`: Explanation of the result
- `confidence`: Confidence score (0.0-1.0)
- `sub_agent_results`: Results from sub-agents if any

**Raises:**
- `RuntimeError`: If bot is not started or has been terminated
- `ValueError`: If `user_prompt` is empty, `max_iterations` < 1, or `max_clarification_rounds` < 0
- `TypeError`: If `clarification_callback` is provided but not callable, or `base_knowledge` is not a list

#### Convenience Methods

##### `get_url() -> str`

Get the current page URL.

```python
current_url = bot.get_url()
```

**Returns:** `str` - Current page URL

**Raises:** `RuntimeError` if bot is not started

##### `get_title() -> str`

Get the current page title.

```python
title = bot.get_title()
```

**Returns:** `str` - Current page title

**Raises:** `RuntimeError` if bot is not started

##### `wait_for_load(timeout: int = 30000, state: str = "networkidle") -> None`

Wait for the page to finish loading.

```python
bot.wait_for_load(timeout=5000, state="domcontentloaded")
```

**Parameters:**
- `timeout` (int): Maximum time to wait in milliseconds (default: 30000)
- `state` (str): Load state to wait for: "load", "domcontentloaded", or "networkidle" (default: "networkidle")

**Raises:**
- `RuntimeError`: If bot is not started
- `ValueError`: If invalid state is provided

##### `screenshot(path: Optional[str] = None, full_page: bool = False) -> bytes`

Take a screenshot of the current page.

```python
# Save to file
bot.screenshot(path="screenshot.png", full_page=True)

# Get as bytes
image_bytes = bot.screenshot(full_page=False)
```

**Parameters:**
- `path` (str, optional): File path to save screenshot. If None, returns bytes
- `full_page` (bool): If True, capture full page. If False, capture viewport only (default: False)

**Returns:** `bytes` - Screenshot image data (if path is None)

**Raises:** `RuntimeError` if bot is not started or screenshot fails

#### Property Accessors

##### `is_started: bool` (read-only)

Check if the bot has been started.

```python
if bot.is_started:
    bot.act("Click button")
```

##### `is_terminated: bool` (read-only)

Check if the bot has been terminated.

```python
if not bot.is_terminated:
    bot.act("Click button")
```

##### `current_url: str` (read-only)

Get the current page URL (same as `get_url()`).

```python
url = bot.current_url
```

**Raises:** `RuntimeError` if bot is not started

##### `session_stats: Dict[str, Any]` (read-only)

Get session statistics.

```python
stats = bot.session_stats
# Returns: {
#     "interaction_count": 10,
#     "url_history": 5,
#     "commands_executed": 8,
#     "current_url": "https://example.com"
# }
```

**Raises:** `RuntimeError` if bot is not started

##### `switch_to_page(page) -> None`

Switch to a different browser tab/page. All bot components will be synchronized to the new page.

```python
new_page = context.new_page()
bot.switch_to_page(new_page)
```

##### `end() -> Optional[str]`

Terminate the bot, stop GIF recording, and prevent any subsequent operations.

```python
gif_path = bot.end()  # Returns path to GIF if recording was enabled
```

**Returns:** Optional[str] - Path to the generated GIF if recording was enabled, None otherwise

### Browser Provider

Create and configure browser instances.

```python
from browser_provider import create_browser_provider, BrowserConfig

# Create with config
browser_config = BrowserConfig(
    headless=False,
    stealth_mode=True,
    viewport_width=1920,
    viewport_height=1080
)

provider = create_browser_provider(browser_config)
page = provider.get_page()

# Use with bot
bot = BrowserVisionBot(browser_provider=provider)
```

### Middleware System

Extend bot behavior with middleware.

```python
from middleware import MiddlewareManager, Middleware, ActionContext
from middlewares import LoggingMiddleware, CachingMiddleware, RetryMiddleware

# Create middleware manager
middleware = MiddlewareManager()

# Add built-in middleware
middleware.use(LoggingMiddleware())
middleware.use(CachingMiddleware(ttl=300))
middleware.use(RetryMiddleware(max_retries=3))

# Create custom middleware
class CustomLoggingMiddleware(Middleware):
    async def before_action(self, context: ActionContext):
        print(f"About to execute: {context.action}")
        context.metadata["start_time"] = time.time()
    
    async def after_action(self, context: ActionContext):
        duration = time.time() - context.metadata["start_time"]
        print(f"Action took {duration:.2f}s")

# Use middleware
bot.middleware = middleware
```

## 🔥 Advanced Features

### Action Ledger

Track and audit all actions with the action ledger:

```python
from action_ledger import ActionLedger

# Action ledger is automatically initialized
# Access it via bot.action_ledger

# After actions, query the ledger
actions = bot.action_ledger.filter_records()
for action in actions:
    print(f"{action.id}: {action.goal} - {action.status}")

# Get actions by status
from action_ledger import ActionStatus
failed = bot.action_ledger.filter_records(status=ActionStatus.FAILED)
successful = bot.action_ledger.filter_records(status=ActionStatus.COMPLETED)
```

### Action Queue

Queue actions for deferred execution:

```python
from action_queue import ActionQueue

queue = ActionQueue()

# Queue actions
queue.enqueue("Click the menu button")
queue.enqueue("Click settings")
queue.enqueue("Toggle dark mode")

# Execute queue
while not queue.is_empty():
    action = queue.dequeue()
    bot.act(action)
```


### Mini Goals System

Mini goals allow you to create trigger-based sub-objectives that activate automatically when specific conditions are met. This is perfect for handling complex UI interactions that require specialized logic, like dropdown menus, multi-step forms, or custom widgets.

#### Two Execution Modes

**Autonomy Mode**: The agent handles the mini goal as a complete sub-task, with full planning and completion evaluation.

**Scripted Mode**: Execute custom Python functions that can interact with the page and ask the agent questions using its current context.

#### Registering Mini Goals

```python
from agent.mini_goal_manager import MiniGoalTrigger, MiniGoalMode

# Example: Handle dropdown selections autonomously
trigger = MiniGoalTrigger(
    action_type="click",
    target_regex="dropdown|select.*field"
)

bot.register_mini_goal(
    trigger=trigger,
    mode=MiniGoalMode.AUTONOMY,
    instruction_override="Select the most appropriate option from this dropdown based on the context"
)

# Example: Handle form validation with a script
def validate_form_handler(context):
    # Ask the agent for validation info
    validation_rules = context.ask_question("What validation rules should be applied to this form?")

    # Interact with the page
    context.bot.page.evaluate("""
        // Custom validation logic
        const inputs = document.querySelectorAll('input');
        inputs.forEach(input => {
            if (!input.checkValidity()) {
                input.style.border = '2px solid red';
            }
        });
    """)

bot.register_mini_goal(
    trigger=MiniGoalTrigger(action_type="click", target_regex="submit.*form"),
    mode=MiniGoalMode.SCRIPTED,
    handler=validate_form_handler
)
```

#### Trigger Types

- **Action Triggers**: Activate when the agent performs specific actions (e.g., clicking a dropdown)
- **Observation Triggers**: Activate when specific content appears on the page
- **Selector Triggers**: Activate when interacting with specific DOM elements

#### Features

- **Recursion Control**: Configurable recursion limit (default: 3) prevents infinite mini goal loops
- **State Isolation**: Mini goals maintain their own completion context separate from the main task
- **Navigation Blocking**: In autonomy mode, navigation actions are blocked to keep focus on the mini goal
- **Question Asking**: Scripted handlers can ask the agent contextual questions using current page state

### Error Handling

Robust error handling with retries and recovery:

```python
from bot_config import BotConfig, ErrorHandlingConfig

# Configure error handling
config = BotConfig(
    error_handling=ErrorHandlingConfig(
        screenshot_on_error=True,
        max_retries=3,
        retry_delay=2.0,
        retry_backoff=2.0,
        abort_on_critical=True
    )
)

# Errors are automatically handled based on configuration
# Screenshots saved to error_screenshots/ directory
```

### Interaction Deduplication

Prevent repeating the same actions:

```python
# Deduplication is automatically enabled
# Control via config:
config = BotConfig(
    execution=ExecutionConfig(
        dedup_mode="auto",  # "auto", "on", or "off"
        dedup_history_quantity=-1  # -1 = unlimited
    )
)

# Manually enable/disable
bot.act("dedup: on")
bot.act("dedup: off")
```

## Creating Custom Conditions

> **Note**: The conditional goal system has been removed. The bot now uses keyword-based goal execution only. Use direct keyword goals like `click:`, `type:`, `press:`, etc. For complex workflows, use agentic mode which handles conditionals automatically.

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    BrowserVisionBot                      │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Vision   │  │    Agent     │  │      Tab         │  │
│  │   System   │  │  Controller  │  │   Management    │  │
│  └────────────┘  └──────────────┘  └────────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Action   │  │  Extraction   │  │   Middleware    │  │
│  │  Executor  │  │   Pipeline    │  │     System      │  │
│  └────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │    Playwright    │
              │     Browser      │
              └──────────────────┘
```

### Component Relationships

- **vision_bot.py**: Main orchestrator, handles `act()`, `extract()`, and `execute_task()`
- **agent/agent_controller.py**: Implements reactive agent loop with completion evaluation
- **action_executor.py**: Low-level action execution with retries and fallbacks
- **session_tracker.py**: Tracks browser state, interactions, and navigation history
- **ai_utils.py**: LLM and vision API calls for planning and analysis
- **tab_management/**: Tab tracking, decision engine, and sub-agent coordination
- **element_detection/**: Element detection and overlay management
- **handlers/**: Specialized handlers for selects, uploads, datetime pickers
- **middlewares/**: Extensible middleware for cross-cutting concerns
- **utils/**: Utilities for logging, parsing, vision, and page operations
- **planner/**: Plan generation for AI-based action planning

### Data Flow

1. **User Request** → `bot.act()` or `bot.execute_task()`
2. **Vision Analysis** → Screenshot + overlay generation
3. **LLM Planning** → Determine action plan
4. **Action Execution** → Execute via ActionExecutor
5. **Session Tracking** → Record interactions and browser state
6. **Completion Check** → Evaluate if task completed (agent mode only)
7. **Result** → Return success/failure with evidence

### Interaction summarization controls

- Completion evaluation (`CompletionContract`) and next-action generation (`ReactiveGoalDeterminer`) now accept configurable interaction history limits.
- Defaults include all recorded interactions; pass `interaction_summary_limit_completion` or `interaction_summary_limit_action` into `AgentController` to cap how many recent interactions are summarized in prompts.

## 📖 Examples

### Example 1: Simple Form Filling

```python
from playwright.sync_api import sync_playwright
from vision_bot import BrowserVisionBot

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_context().new_page()
    
    bot = BrowserVisionBot(page=page)
    bot.start()
    
    # Navigate and fill form
    page.goto("https://example.com/contact")
    bot.act("Type 'John Doe' into the name field")
    bot.act("Type 'john@example.com' into the email field")
    bot.act("Type 'Hello, I need help' into the message field")
    bot.act("Click the submit button")
    
    browser.close()
```

### Example 2: Data Extraction

```python
from pydantic import BaseModel
from typing import List

class Article(BaseModel):
    title: str
    author: str
    date: str
    summary: str

# Navigate to news site
page.goto("https://news.example.com")

# Extract structured data
articles = bot.extract(
    "Extract all articles on this page",
    output_format="structured",
    model_schema=List[Article]
)

for article in articles:
    print(f"{article.title} by {article.author}")
```

### Example 3: Multi-Tab Workflow

```python
# Agent automatically manages tabs
result = bot.execute_task(
    user_prompt="""
    Research the following topics in parallel:
    1. Latest AI developments
    2. Climate change news
    3. Stock market trends
    
    Extract key points from each topic.
    """,
    max_iterations=30
)

# Results from sub-agents in different tabs
for sub_result in result.sub_agent_results:
    print(f"Sub-agent: {sub_result.agent_id}")
    print(f"Data: {sub_result.extracted_data}")
```

### Example 4: Agentic Mode for Complex Tasks

```python
# Use execute_task for complex workflows that require conditional logic
result = bot.execute_task(
    "Navigate to the login page, fill in username 'user' and password 'pass', then click login"
)

# execute_task handles conditionals automatically
result = bot.execute_task(
    "Go through all pages of search results and extract each product name and price"
)

# Limit how much interaction history is summarized in LLM prompts
result = bot.execute_task(
    "Submit the contact form",
    interaction_summary_limit_completion=50,
    interaction_summary_limit_action=30,
)
```

### Example 5: Complete Workflow

```python
from browser_vision_bot import BotConfig, create_browser_provider

# Configure bot
config = BotConfig.production()
browser_provider = create_browser_provider(config.browser)

# Initialize
bot = BrowserVisionBot(config=config, browser_provider=browser_provider)
bot.start()

# Navigate
bot.page.goto("https://example.com")

# Complex task with execute_task
result = bot.execute_task(
    user_prompt="Search for 'Python tutorials', open the first 3 results, and extract their titles",
    max_iterations=20
)

if result.success:
    print("Extracted titles:")
    for title in result.extracted_data.values():
        print(f"  - {title}")

# Cleanup
gif_path = bot.end()
if gif_path:
    print(f"Recording saved to: {gif_path}")
```

## 🛠️ Development

```
browser-vision-bot/
├── vision_bot.py              # Main bot implementation
├── action_executor.py         # Action execution engine
├── ai_utils.py               # LLM/vision API utilities
├── bot_config.py             # Configuration models
├── browser_provider.py       # Browser management
├── action_ledger.py         # Action tracking
├── action_queue.py           # Action queuing
├── interaction_deduper.py    # Deduplication
├── middleware.py             # Middleware system
├── error_handling.py         # Error handling
├── gif_recorder.py           # GIF recording
├── vision_utils.py           # Vision utilities
├── agent/                    # Agent system
│   ├── agent_controller.py   # Main agent loop
│   ├── agent_context.py       # Agent context
│   ├── agent_result.py        # Agent results
│   ├── completion_contract.py # Completion evaluation
│   ├── reactive_goal_determiner.py # Next action determination
│   ├── stuck_detector.py      # Stuck detection
│   └── sub_agent_controller.py # Sub-agent management
├── tab_management/           # Tab orchestration
│   ├── tab_manager.py        # Tab tracking
│   ├── tab_decision_engine.py # Tab decisions
│   └── tab_info.py           # Tab metadata
├── element_detection/        # Element detection
│   ├── element_detector.py   # Element detection
│   └── overlay_manager.py    # Overlay management
├── handlers/                 # Specialized handlers
│   ├── select_handler.py     # Dropdown/select handling (native & custom)
│   ├── upload_handler.py     # File upload handling
│   └── datetime_handler.py   # Date/time handling
├── middlewares/              # Built-in middleware
│   ├── logging_middleware.py
│   ├── caching.py
│   ├── retry.py
│   ├── cost_tracking.py
│   ├── metrics.py
│   ├── human_in_loop.py
│   └── error_handling_middleware.py
├── models/                   # Data models
│   ├── core_models.py        # Core models
│   └── intent_models.py      # Intent models
├── utils/                    # Utilities
│   ├── bot_logger.py         # Logging
│   ├── event_logger.py       # Event logging
│   ├── intent_parsers.py     # Intent parsing
│   ├── page_utils.py         # Page utilities
│   ├── selector_utils.py    # Selector utilities
│   ├── vision_resolver.py    # Vision resolution
│   └── ...
├── planner/                  # Planning system
│   └── plan_generator.py     # Plan generation
├── session_tracker.py        # Session state tracking
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
└── requirements.txt          # Dependencies
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Notes
- Scroll actions now normalize scroll positions to integers to avoid validation errors during PageInfo checks.
- Upload handling: when no file path is provided, the upload control is clicked and execution waits for the user to pick a file (press Enter to resume).
- If an upload path is provided but doesn’t exist, the upload handler now falls back to the manual picker flow (click + wait for user to select).

# Run specific test file
pytest tests/unit/test_tab_manager.py -v

# Test select field handling (native and custom patterns)
pytest tests/integration/test_select_handler_fixture.py -v
pytest tests/integration/test_selector_coordinates.py -v  # Test selector coordinate resolution
```

### Select fixture for manual testing

Open `tests/integration/select_fixtures.html` in a browser to exercise native and custom dropdown scenarios. One simple way is to serve the integration folder locally:

```bash
python -m http.server 8000 -d tests/integration
# visit http://localhost:8000/select_fixtures.html
```

### Environment Variables

```env
# Required: AI API Keys
GEMINI_API_KEY=your_gemini_api_key
# OR
OPENAI_API_KEY=your_openai_api_key

# Optional: Logging
LOG_LEVEL=INFO
DEBUG_MODE=true

# Optional: Browser
HEADLESS=false
BROWSER_TIMEOUT=30000
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Playwright](https://playwright.dev/)
- Powered by [Google Gemini](https://ai.google.dev/) and [OpenAI](https://openai.com/)
- Uses [Pydantic](https://pydantic.dev/) for data validation

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**BrowserVisionBot** - Intelligent web automation powered by AI vision 🤖✨
