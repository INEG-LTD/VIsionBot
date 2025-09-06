# Creating New Conditions for Conditional Goals

This guide explains how to create new conditions for the conditional goal system. Conditions are the building blocks that determine which sub-goals to execute in conditional goals like `IfGoal`.

## Overview

A `Condition` consists of:
- **Type**: The category of condition (environment state, computational, user-defined)
- **Description**: Human-readable description of what the condition checks
- **Evaluator**: A function that takes a `GoalContext` and returns `True`/`False`
- **Confidence Threshold**: Minimum confidence level for the condition (default: 0.8)

## Condition Types

### 1. Environment State Conditions
Test various aspects of the browser/page state:
- Element existence, visibility, or properties
- Page content, URLs, or navigation state
- Form field states or user interactions
- DOM structure or element counts

### 2. Computational Conditions
Perform calculations or logic operations:
- Date/time checks (weekdays, business hours, etc.)
- Mathematical expressions
- Data processing or validation
- Business logic evaluations

### 3. User-Defined Conditions
Custom conditions for specific use cases:
- External API calls
- File system checks
- Database queries
- Custom business rules

## Creating a New Condition

### Step 1: Choose the Condition Type

```python
from goals.base import ConditionType, create_environment_condition, create_computational_condition, create_user_defined_condition
```

### Step 2: Implement the Evaluator Function

The evaluator function must:
- Take a `GoalContext` as its only parameter
- Return a boolean (`True`/`False`)
- Handle exceptions gracefully

```python
def my_condition_evaluator(context: GoalContext) -> bool:
    """
    Evaluate whether the condition is met.
    
    Args:
        context: Complete browser state and interaction history
        
    Returns:
        True if condition is met, False otherwise
    """
    try:
        # Your condition logic here
        # Access context.current_state, context.page_reference, etc.
        return True  # or False based on your logic
    except Exception as e:
        # Log error and return False for safety
        print(f"Error in condition evaluator: {e}")
        return False
```

### Step 3: Create the Condition

Use the appropriate factory function:

```python
# For environment state conditions
condition = create_environment_condition(
    description="My custom condition",
    evaluator=my_condition_evaluator,
    confidence_threshold=0.9  # Optional, defaults to 0.8
)

# For computational conditions
condition = create_computational_condition(
    description="My computational condition",
    evaluator=my_condition_evaluator
)

# For user-defined conditions
condition = create_user_defined_condition(
    description="My user-defined condition",
    evaluator=my_condition_evaluator
)
```

## Common Patterns

### Environment State Conditions

#### Check Element Properties
```python
def element_has_text_condition(selector: str, expected_text: str) -> Condition:
    """Check if an element contains specific text"""
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        try:
            element = context.page_reference.query_selector(selector)
            if not element:
                return False
            actual_text = element.text_content() or ""
            return expected_text.lower() in actual_text.lower()
        except Exception:
            return False
    
    return create_environment_condition(
        f"Element '{selector}' contains text '{expected_text}'",
        evaluator
    )
```

#### Check Page State
```python
def page_loaded_condition() -> Condition:
    """Check if page has finished loading"""
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        try:
            # Check if page is in a loaded state
            return context.page_reference.evaluate("document.readyState") == "complete"
        except Exception:
            return False
    
    return create_environment_condition(
        "Page has finished loading",
        evaluator
    )
```

#### Check Form State
```python
def form_valid_condition(form_selector: str) -> Condition:
    """Check if a form is valid"""
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        try:
            form = context.page_reference.query_selector(form_selector)
            if not form:
                return False
            return form.evaluate("form.checkValidity()")
        except Exception:
            return False
    
    return create_environment_condition(
        f"Form '{form_selector}' is valid",
        evaluator
    )
```

### Computational Conditions

#### Date/Time Checks
```python
def is_business_hours_condition() -> Condition:
    """Check if current time is during business hours (9 AM - 5 PM)"""
    def evaluator(context: GoalContext) -> bool:
        import datetime
        now = datetime.datetime.now()
        return 9 <= now.hour < 17 and now.weekday() < 5
    
    return create_computational_condition(
        "Current time is during business hours",
        evaluator
    )
```

#### Mathematical Expressions
```python
def percentage_condition(current: int, total: int, threshold: float) -> Condition:
    """Check if current/total percentage meets threshold"""
    def evaluator(context: GoalContext) -> bool:
        if total == 0:
            return False
        percentage = (current / total) * 100
        return percentage >= threshold
    
    return create_computational_condition(
        f"Percentage ({current}/{total}) >= {threshold}%",
        evaluator
    )
```

### User-Defined Conditions

#### External API Check
```python
def api_available_condition(api_url: str) -> Condition:
    """Check if an external API is available"""
    def evaluator(context: GoalContext) -> bool:
        try:
            import requests
            response = requests.get(api_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    return create_user_defined_condition(
        f"API at '{api_url}' is available",
        evaluator
    )
```

#### File System Check
```python
def file_exists_condition(file_path: str) -> Condition:
    """Check if a file exists"""
    def evaluator(context: GoalContext) -> bool:
        import os
        return os.path.exists(file_path)
    
    return create_user_defined_condition(
        f"File '{file_path}' exists",
        evaluator
    )
```

## Using Conditions in Goals

Once you have a condition, use it in conditional goals:

```python
from goals.if_goal import IfGoal

# Create your condition
condition = my_condition_evaluator()

# Create sub-goals
success_goal = SomeGoal("Action when condition is true")
fail_goal = SomeGoal("Action when condition is false")

# Create the conditional goal
if_goal = IfGoal(condition, success_goal, fail_goal)

# Use it like any other goal
result = if_goal.evaluate(context)
```

## Best Practices

### 1. Error Handling
Always wrap your evaluator logic in try-catch blocks:

```python
def safe_evaluator(context: GoalContext) -> bool:
    try:
        # Your logic here
        return True
    except Exception as e:
        print(f"Condition evaluation failed: {e}")
        return False  # Default to False for safety
```

### 2. Null Checks
Check for required objects before using them:

```python
def safe_evaluator(context: GoalContext) -> bool:
    if not context.page_reference:
        return False
    if not context.current_state:
        return False
    # Continue with your logic
```

### 3. Descriptive Names
Use clear, descriptive names for your conditions:

```python
# Good
condition = element_visible_condition("#submit-button")

# Bad
condition = check_condition("#btn")
```

### 4. Confidence Thresholds
Set appropriate confidence thresholds:

```python
# High confidence for simple checks
condition = create_environment_condition(
    "Element exists",
    evaluator,
    confidence_threshold=0.95
)

# Lower confidence for complex checks
condition = create_user_defined_condition(
    "External API responds",
    evaluator,
    confidence_threshold=0.7
)
```

### 5. Documentation
Document your conditions thoroughly:

```python
def complex_business_condition() -> Condition:
    """
    Check if the current user has permission to access the admin panel.
    
    This condition:
    1. Verifies the user is logged in
    2. Checks their role in the session
    3. Validates against the current page context
    
    Returns:
        True if user has admin access, False otherwise
    """
    def evaluator(context: GoalContext) -> bool:
        # Implementation here
        pass
    
    return create_user_defined_condition(
        "User has admin panel access",
        evaluator
    )
```

## Testing Your Conditions

Create test cases for your conditions:

```python
def test_my_condition():
    """Test the custom condition"""
    # Create test context
    context = create_mock_context()
    
    # Create condition
    condition = my_condition_evaluator()
    
    # Test evaluation
    result = condition.evaluator(context)
    assert isinstance(result, bool)
    
    # Test with different contexts
    # ... more test cases
```

## Adding to Condition Utils

If your condition is reusable, add it to `goals/condition_utils.py`:

```python
def my_reusable_condition(param1: str, param2: int, description: Optional[str] = None) -> Condition:
    """Reusable condition that can be used across the system"""
    if description is None:
        description = f"My condition with {param1} and {param2}"
    
    def evaluator(context: GoalContext) -> bool:
        # Your implementation
        pass
    
    return create_environment_condition(description, evaluator)
```

This makes your condition available throughout the system and allows it to be discovered by the AI condition parser.

## Conclusion

Creating new conditions is straightforward once you understand the pattern. Remember to:
- Choose the right condition type
- Implement robust error handling
- Use descriptive names and documentation
- Test your conditions thoroughly
- Consider reusability and add to condition_utils.py when appropriate

The conditional goal system is designed to be extensible, so don't hesitate to create conditions for your specific use cases!
