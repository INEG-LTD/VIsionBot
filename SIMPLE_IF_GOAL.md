# IfGoal - Vision-Based Conditional Execution

The IfGoal now uses a clean, vision-based approach for yes/no questions instead of the complex condition system.

## Overview

Instead of complex condition parsing and evaluation, IfGoal uses the vision model to answer simple yes/no questions about the current page state.

## How It Works

1. **Question Conversion**: Converts condition text to natural yes/no questions
2. **Vision Analysis**: Takes a screenshot and asks the vision model the question
3. **Action Selection**: Executes success or fail action based on the answer

## Usage Examples

### Basic Usage

```python
# Old complex way (removed)
# if: a cookie banner is visible then: click: accept cookies

# New simple way
bot.act("if: a cookie banner is visible then: click: accept cookies")
```

### With Fail Action

```python
bot.act("if: login button is visible then: click: login button else: type: username into: username field")
```

### Without Fail Action

```python
bot.act("if: cookie banner is visible then: click: accept cookies")
# If no cookie banner, nothing happens
```

## Question Conversion

The system automatically converts condition text to natural questions:

| Original Condition | Converted Question |
|-------------------|-------------------|
| `a button is visible` | `Is a button visible?` |
| `page contains login` | `Does the page contain login?` |
| `on login page` | `Are we on the login page?` |
| `cookie banner visible` | `Is cookie banner visible?` |

## Screenshot Types

- **viewport** (default): Takes screenshot of visible area
- **full_page**: Takes screenshot of entire page

## Benefits

### ✅ **Simplified**
- No complex condition parsing
- No predicate evaluation
- Just natural language questions

### ✅ **Reliable**
- Uses vision model for accurate page analysis
- Handles dynamic content well
- Works with any page layout

### ✅ **Flexible**
- Supports any yes/no question
- Easy to understand and debug
- Natural language conditions

### ✅ **Fast**
- Single vision call per condition
- No complex expression evaluation
- Direct action execution

## Implementation Details

### IfGoal Class

```python
class IfGoal(BaseGoal):
    def __init__(self, question: str, success_action: str, 
                 fail_action: Optional[str] = None, 
                 screenshot_type: str = "viewport"):
        # ...
```

### Evaluation Process

1. **Screenshot**: Captures page based on `screenshot_type`
2. **Vision Query**: Asks `question` to vision model
3. **Action Selection**: 
   - If answer is "yes" → execute `success_action`
   - If answer is "no" → execute `fail_action` (if provided)
4. **Result**: Returns action to execute or completion status

### Integration

The IfGoal is automatically used when parsing `if:` statements:

```python
# This automatically creates an IfGoal
bot.act("if: login button visible then: click: login button")
```

## Comparison: Old vs New

### Old Complex System
```python
# Complex condition parsing
condition = compile_nl_to_expr("a button is visible")
predicate = create_predicate_condition(condition)
if_goal = IfGoal(condition, success_goal, fail_goal)
result = if_goal.evaluate(context)
```

### New Simple System
```python
# Simple vision-based question
if_goal = IfGoal(
    question="Is a button visible?",
    success_action="click: the button",
    fail_action="scroll: down"
)
result = if_goal.evaluate(context)
```

## Error Handling

- **Vision errors**: Gracefully handled with fallback
- **Screenshot errors**: Caught and reported
- **Action errors**: Don't affect the condition evaluation

## Debugging

The system provides clear debug output:

```
🔀 Converted condition to question: 'Is a cookie banner visible?'
🔍 IfGoal: 'Is a cookie banner visible?'
   🎯 Vision answer: YES
   ✅ Executing success action: click: accept cookies
🔀 IfGoal resolved to action: click: accept cookies
```

## Best Practices

1. **Use Natural Language**: Write conditions as you would ask them
2. **Be Specific**: "login button visible" vs "button visible"
3. **Test Conditions**: Verify questions work with your pages
4. **Handle Edge Cases**: Provide fail actions when needed

## Migration

The old complex IfGoal system is replaced automatically. No changes needed to existing code:

```python
# This still works exactly the same
bot.act("if: cookie banner visible then: click: accept cookies")
```

But now it uses the much simpler and more reliable vision-based approach!
