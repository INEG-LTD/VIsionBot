# BrowserVisionBot

A powerful, vision-based web automation framework that uses AI to interact with web pages like a human would. BrowserVisionBot combines computer vision, large language models (LLMs), and Playwright to create intelligent automation agents that can understand and interact with any web interface.

## 🌟 Key Features

- **Vision-Based Automation**: Uses AI vision models to understand web pages visually, not just through DOM inspection
- **Intelligent Agent System**: Autonomous agents that can plan, execute, and adapt to achieve complex goals
- **Multi-Tab Management**: Sophisticated tab orchestration with sub-agent support for parallel workflows
- **Flexible Action System**: Supports clicks, typing, form filling, file uploads, navigation, and custom actions
- **Data Extraction**: Extract structured data from web pages using natural language prompts
- **Stealth Capabilities**: Built-in stealth features to avoid bot detection
- **Middleware System**: Extensible middleware for logging, caching, error handling, and custom behaviors
- **Smart Error Recovery**: Automatic retry logic, fallback strategies, and stuck detection
- **Configuration-Driven**: Type-safe configuration using Pydantic models

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

```python
from playwright.sync_api import sync_playwright
from vision_bot import BrowserVisionBot
from bot_config import BotConfig

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
    
    browser.close()
```

### Using Configuration

```python
from bot_config import BotConfig, ModelConfig, ExecutionConfig
from browser_provider import create_browser_provider

# Create a configuration
config = BotConfig(
    model=ModelConfig(agent_model="gpt-5-mini"),
    execution=ExecutionConfig(fast_mode=True, max_attempts=15)
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
result = bot.agentic_mode(
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
- **Completion Evaluation**: LLM-based assessment of whether the goal is achieved
- **Adaptive Planning**: Dynamically adjusts strategy based on page state and history
- **Stuck Detection**: Identifies when the agent is stuck in a loop and takes corrective action

### Tab Management

Sophisticated multi-tab orchestration:

- **Tab Tracking**: Automatic registration and metadata tracking for all tabs
- **Tab Decisions**: LLM-based decisions on when to switch, close, or create tabs
- **Sub-Agents**: Spawn independent agents in separate tabs for parallel workflows
- **Tab Synchronization**: Keeps all components synchronized when switching tabs

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
from bot_config import BotConfig, ModelConfig, ExecutionConfig, CacheConfig

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
        fast_mode=False,
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
    )
)
```

### Preset Configurations

```python
# Fast mode - optimized for speed
config = BotConfig.fast()

# Debug mode - with GIF recording and verbose logging
config = BotConfig.debug()

# Production mode - balanced for reliability
config = BotConfig.production()

# Minimal - all defaults
config = BotConfig.minimal()
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

#### Core Methods

##### `start()`

Initialize the bot and register the initial page.

```python
bot.start()
```

##### `act(goal_description, **kwargs)`

Execute a single action based on natural language description.

```python
result = bot.act("Click the login button")
result = bot.act("Type 'username' into the username field")
result = bot.act("Press Enter")
```

**Parameters:**
- `goal_description` (str): Natural language description of the action
- `additional_context` (str, optional): Extra context for the action
- `interpretation_mode` (str, optional): "keyword" or "vision"
- `max_attempts` (int, optional): Maximum retry attempts

**Returns:** `GoalResult` with success status, confidence, and reasoning

##### `extract(prompt, **kwargs)`

Extract data from the current page.

```python
# Extract text
title = bot.extract("What is the page title?")

# Extract structured data
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    description: str

product = bot.extract(
    "Extract product information",
    output_format="structured",
    model_schema=Product
)
```

**Parameters:**
- `prompt` (str): What to extract
- `output_format` (str): "text", "json", or "structured"
- `model_schema` (Type[BaseModel], optional): Pydantic model for structured output
- `scope` (str): "viewport", "full_page", or "element"
- `confidence_threshold` (float): Minimum confidence (0.0-1.0)

**Returns:** Extracted data in requested format

##### `agentic_mode(user_prompt, **kwargs)`

Run autonomous agent to complete a complex task.

```python
result = bot.agentic_mode(
    user_prompt="Research the top 3 AI companies and extract their names",
    max_iterations=20,
    base_knowledge="Focus on companies founded after 2015"
)

if result.success:
    print("Extracted data:", result.extracted_data)
    print("Sub-agent results:", result.sub_agent_results)
```

**Parameters:**
- `user_prompt` (str): High-level task description
- `max_iterations` (int): Maximum agent iterations
- `base_knowledge` (str, optional): Domain knowledge or constraints
- `agent_context` (AgentContext, optional): For sub-agents

**Returns:** `AgentResult` with success, extracted data, and orchestration details

##### `switch_to_page(page)`

Switch to a different browser tab/page.

```python
new_page = context.new_page()
bot.switch_to_page(new_page)
```

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
from middleware import MiddlewareManager
from middlewares import LoggingMiddleware, CachingMiddleware, RetryMiddleware

# Create middleware manager
middleware = MiddlewareManager()

# Add middleware
middleware.use(LoggingMiddleware())
middleware.use(CachingMiddleware(ttl=300))
middleware.use(RetryMiddleware(max_retries=3))

# Use with bot
bot = BrowserVisionBot(page=page)
bot.middleware_manager = middleware
```

## 🔥 Advanced Features

### Command Ledger

Track and audit all actions with the command ledger:

```python
from command_ledger import CommandLedger

ledger = CommandLedger()

# Enable ledger
bot.command_ledger = ledger

# After actions, query the ledger
commands = ledger.get_all_commands()
for cmd in commands:
    print(f"{cmd.command_id}: {cmd.command_text} - {cmd.status}")

# Get commands by status
failed = ledger.get_commands_by_status("failed")
successful = ledger.get_commands_by_status("completed")
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

### Post-Action Hooks

Execute callbacks after actions:

```python
def log_action(context):
    print(f"Action completed: {context.action}")
    print(f"Success: {context.result.success}")

def take_screenshot(context):
    if not context.result.success:
        context.page.screenshot(path=f"error_{context.action_id}.png")

# Register hooks
bot.register_post_action_hook(log_action)
bot.register_post_action_hook(take_screenshot)
```

### Sub-Agents

Spawn independent agents for parallel workflows:

```python
# Agent mode automatically manages sub-agents
result = bot.agentic_mode(
    user_prompt="Research 5 different topics and compile a report",
    max_iterations=30
)

# Sub-agents are spawned automatically for parallel research
# Results are aggregated in result.sub_agent_results
```

### Focus Management

Track and manage element focus:

```python
from focus_manager import FocusManager

focus_mgr = FocusManager(page)

# Set focus context
focus_mgr.set_focus_context("login_form")

# Get focused elements
elements = focus_mgr.get_focused_elements()

# Clear focus
focus_mgr.clear_focus()
```

### Error Handling

Robust error handling with retries and recovery:

```python
from error_handling import ErrorHandler, ErrorRecoveryStrategy

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

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    BrowserVisionBot                      │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Vision   │  │    Agent     │  │      Tab       │  │
│  │   System   │  │  Controller  │  │   Management   │  │
│  └────────────┘  └──────────────┘  └────────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Action   │  │  Extraction  │  │   Middleware   │  │
│  │  Executor  │  │   Pipeline   │  │     System     │  │
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

- **vision_bot.py**: Main orchestrator, handles `act()`, `extract()`, and `agentic_mode()`
- **agent/agent_controller.py**: Implements reactive agent loop with completion evaluation
- **action_executor.py**: Low-level action execution with retries and fallbacks
- **ai_utils.py**: LLM and vision API calls for planning and analysis
- **tab_management/**: Tab tracking, decision engine, and sub-agent coordination
- **element_detection/**: Element detection and overlay management
- **handlers/**: Specialized handlers for selects, uploads, datetime pickers
- **middlewares/**: Extensible middleware for cross-cutting concerns
- **utils/**: Utilities for logging, parsing, vision, and page operations

### Data Flow

1. **User Request** → `bot.act()` or `bot.agentic_mode()`
2. **Vision Analysis** → Screenshot + overlay generation
3. **LLM Planning** → Determine action plan
4. **Action Execution** → Execute via ActionExecutor
5. **Completion Check** → Evaluate if goal achieved
6. **Result** → Return success/failure with evidence

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
result = bot.agentic_mode(
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

### Example 4: Custom Middleware

```python
from middleware import Middleware, ActionContext

class CustomLoggingMiddleware(Middleware):
    async def before_action(self, context: ActionContext):
        print(f"About to execute: {context.action}")
        context.metadata["start_time"] = time.time()
    
    async def after_action(self, context: ActionContext):
        duration = time.time() - context.metadata["start_time"]
        print(f"Action took {duration:.2f}s")

# Use middleware
bot.middleware_manager.use(CustomLoggingMiddleware())
```

## 🛠️ Development

### Project Structure

```
browser-vision-bot/
├── vision_bot.py              # Main bot implementation
├── action_executor.py         # Action execution engine
├── ai_utils.py               # LLM/vision API utilities
├── bot_config.py             # Configuration models
├── browser_provider.py       # Browser management
├── command_ledger.py         # Command tracking
├── action_queue.py           # Action queuing
├── focus_manager.py          # Focus management
├── interaction_deduper.py    # Deduplication
├── middleware.py             # Middleware system
├── error_handling.py         # Error handling
├── gif_recorder.py           # GIF recording
├── vision_utils.py           # Vision utilities
├── agent/                    # Agent system
│   ├── agent_controller.py   # Main agent loop
│   ├── agent_context.py      # Agent context
│   ├── agent_result.py       # Agent results
│   ├── completion_contract.py # Completion evaluation
│   ├── reactive_goal_determiner.py # Goal determination
│   ├── stuck_detector.py     # Stuck detection
│   └── sub_agent_controller.py # Sub-agent management
├── tab_management/           # Tab orchestration
│   ├── tab_manager.py        # Tab tracking
│   ├── tab_decision_engine.py # Tab decisions
│   └── tab_info.py           # Tab metadata
├── element_detection/        # Element detection
│   ├── element_detector.py   # Element detection
│   └── overlay_manager.py    # Overlay management
├── handlers/                 # Specialized handlers
│   ├── select_handler.py     # Dropdown handling
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
│   ├── selector_utils.py     # Selector utilities
│   ├── vision_resolver.py    # Vision resolution
│   └── ...
├── planner/                  # Planning system
│   └── plan_generator.py     # Plan generation
├── goals/                    # Goal evaluators
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

# Run specific test file
pytest tests/unit/test_tab_manager.py -v
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
