# Terminology Definitions

This document defines the core terminology used throughout the browser-vision-bot project. All code, comments, and documentation should use these terms consistently.

## Core Terms

### Task
**Definition**: A high-level, autonomous work item that the agent executes from start to finish.

**Characteristics**:
- Represents a complete user request
- May contain multiple goals
- Executed via `execute_task(user_prompt)`
- Returns `AgentResult` with success status and extracted data

**Examples**:
- "go to elon wikipedia"
- "search for python tutorials"
- "navigate to amazon.com, search for 'laptop', extract the first product name and price"

**Code References**:
- `execute_task()` method in `BrowserVisionBot`
- `TaskResult` class
- `run_execute_task()` method in `AgentController`

---

### Goal
**Definition**: A sub-objective within a task that specifies what needs to be achieved at a specific point in execution.

**Characteristics**:
- Part of a larger task
- Specifies a single objective (e.g., clicking a button, typing text, navigating)
- May be expressed as a keyword-formatted string (e.g., "click: login button")
- Can be tracked and evaluated for completion

**Examples**:
- "click: wikipedia link" (within task "go to elon wikipedia")
- "type: username in username field" (within task "log into account")
- "scroll: down" (within task "find product information")

**Code References**:
- `goal_description` parameter in `act()` method
- `Goal` model in `models/core_models.py`
- `ReactiveGoalDeterminer` class
- `CompletionContract` evaluates goal completion

---

### Action
**Definition**: An atomic unit of work that can be executed by the bot. The actual execution step that implements a goal.

**Characteristics**:
- Represents a single, executable operation
- Has a specific type (click, type, scroll, press, etc.)
- Contains all parameters needed for execution (coordinates, text, etc.)
- Returns `ActionResult` with success status

**Examples**:
- Clicking at coordinates (x: 100, y: 200)
- Typing "username" into an input field
- Scrolling down 500 pixels

**Code References**:
- `ActionType` enum (CLICK, TYPE, SCROLL, PRESS, etc.)
- `ActionStep` model (contains action type and parameters)
- `ActionResult` class (result of executing an action)
- `NextAction` model (determined by ReactiveGoalDeterminer)
- `ActionExecutor` class
- `ActionQueue` class

**Relationship to Goal**:
- A goal (e.g., "click: login button") is translated into one or more actions (e.g., ActionStep with type=CLICK, coordinates, etc.)

---

### Keyword
**Definition**: The prefix/verb that identifies the type of action to perform, formatted with a colon.

**Characteristics**:
- Always ends with a colon (`:`)
- Specifies the action type
- Part of the goal description format
- Used for parsing and routing to appropriate handlers

**Examples**:
- `"click:"` - for clicking elements
- `"type:"` - for typing text
- `"scroll:"` - for scrolling
- `"press:"` - for keyboard presses
- `"navigate:"` - for navigation
- `"extract:"` - for data extraction

**Code References**:
- `parse_keyword_command()` in `utils/intent_parsers.py`
- `VALID_COMMANDS` set in `NextAction` model
- Keyword parsing logic in `vision_bot.py`

**Format**:
- Goals use keyword format: `"keyword: description"`
- Example: `"click: login button"` where `"click:"` is the keyword and `"login button"` is the description

---

## Terminology Hierarchy

```
Task
  └── Goal 1 ("click: wikipedia link")
      └── Action (ActionStep: CLICK at coordinates)
  └── Goal 2 ("type: search term")
      └── Action (ActionStep: TYPE "search term" in input field)
  └── Goal 3 ("press: Enter")
      └── Action (ActionStep: PRESS Enter key)
```

## Usage Guidelines

1. **When describing user input**: Use "goal" (e.g., "the user's goal is to click the button")
2. **When describing execution**: Use "action" (e.g., "executing the click action")
3. **When describing parsing**: Use "keyword" (e.g., "parsing the click: keyword")
4. **When describing high-level work**: Use "task" (e.g., "the task is to log in")

## Code Examples

```python
# Task level
result = bot.execute_task("go to elon wikipedia")

# Goal level (within task execution)
bot.act("click: wikipedia link")  # goal_description parameter

# Action level (internal execution)
action_step = ActionStep(
    action=ActionType.CLICK,
    x=100,
    y=200
)
result = executor.execute(action_step)

# Keyword parsing
keyword, payload, helper = parse_keyword_command("click: login button")
# keyword = "click"
# payload = "login button"
```

