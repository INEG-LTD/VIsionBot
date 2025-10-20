"""
Utilities for creating and evaluating conditions in conditional goals.
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional

from ai_utils import generate_text
from .base import Condition, GoalContext, create_environment_condition, create_computational_condition


# =============================================================================
# Built-in Condition Evaluators
# =============================================================================

def element_exists_condition(selector: str, description: Optional[str] = None) -> Condition:
    """Check if an element exists on the page using AI vision analysis"""
    if description is None:
        description = f"Element with selector '{selector}' exists"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            print("🔍 DEBUG: element_exists_condition - No page reference")
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to analyze if the element exists
            prompt = f"""
Analyze the screenshot and determine if the requested element exists on the page.

Element to look for: {selector}
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Determine if the element described exists on the page (even if not visible)
3. Consider that the element might be:
   - A button, link, or interactive element
   - Part of a form or modal
   - A banner, popup, or overlay
   - Any other UI element that matches the description
4. The element might exist but be hidden, off-screen, or in a collapsed state

Return only "true" if the element exists on the page, or "false" if it doesn't exist at all.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and determine if the requested element exists on the page. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: element_exists_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for element existence: {response}")
                    return False
            else:
                print("⚠️ No AI response for element existence check")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: element_exists_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


def element_visible_condition(selector: str, description: Optional[str] = None) -> Condition:
    """Check if an element is visible on the page using AI vision analysis"""
    if description is None:
        description = f"Element with selector '{selector}' is visible"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            print("🔍 DEBUG: element_visible_condition - No page reference")
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to analyze if the element is visible
            prompt = f"""
Analyze the screenshot and determine if the requested element is visible on the page.

Element to look for: {selector}
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Determine if the element described is visible on the page
3. Consider that the element might be:
   - A button, link, or interactive element
   - Part of a form or modal
   - A banner, popup, or overlay
   - Any other UI element that matches the description

Return only "true" if the element is clearly visible, or "false" if it's not visible or not present.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and determine if the requested element is visible. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: element_visible_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for element visibility: {response}")
                    return False
            else:
                print("⚠️ No AI response for element visibility check")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: element_visible_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


def text_contains_condition(text: str, description: Optional[str] = None) -> Condition:
    """Check if the page contains specific text using AI vision analysis"""
    if description is None:
        description = f"Page contains text '{text}'"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to analyze if the text is present
            prompt = f"""
Analyze the screenshot and determine if the requested text is present on the page.

Text to look for: "{text}"
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Search for the exact text or similar text on the page
3. Consider that the text might be:
   - In buttons, links, or labels
   - In form fields or placeholders
   - In headings, paragraphs, or other content
   - Part of error messages or notifications
4. Be flexible with exact matching - look for the meaning/intent of the text

Return only "true" if the text (or its meaning) is present on the page, or "false" if it's not found.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and determine if the requested text is present on the page. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: text_contains_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for text search: {response}")
                    return False
            else:
                print("⚠️ No AI response for text search")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: text_contains_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


def url_contains_condition(url_fragment: str, description: Optional[str] = None) -> Condition:
    """Check if the current URL contains a specific fragment"""
    if description is None:
        description = f"URL contains '{url_fragment}'"
    
    def evaluator(context: GoalContext) -> bool:
        return url_fragment.lower() in context.current_state.url.lower()
    
    return create_environment_condition(description, evaluator)


def form_field_filled_condition(selector: str, description: Optional[str] = None) -> Condition:
    """Check if a form field has a value using AI vision analysis"""
    if description is None:
        description = f"Form field '{selector}' is filled"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to analyze if the form field is filled
            prompt = f"""
Analyze the screenshot and determine if the specified form field has a value.

Form field to check: {selector}
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Find the form field described
3. Check if the field contains any text or value
4. Consider that the field might be:
   - A text input, textarea, or select dropdown
   - Part of a form or modal
   - Pre-filled or empty
5. A field is considered "filled" if it contains any visible text or selected value

Return only "true" if the form field has a value, or "false" if it's empty.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and determine if the specified form field has a value. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: form_field_filled_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for form field check: {response}")
                    return False
            else:
                print("⚠️ No AI response for form field check")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: form_field_filled_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


def count_elements_condition(selector: str, expected_count: int, description: Optional[str] = None) -> Condition:
    """Check if there are exactly N elements matching the selector using AI vision analysis"""
    if description is None:
        description = f"Exactly {expected_count} elements with selector '{selector}'"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to count the elements
            prompt = f"""
Analyze the screenshot and count how many elements match the description.

Elements to count: {selector}
Expected count: {expected_count}
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Count all elements that match the description
3. Consider that the elements might be:
   - Buttons, links, or interactive elements
   - Form fields, inputs, or controls
   - Images, icons, or visual elements
   - Any other UI elements that match the description
4. Count only visible elements on the current view

Return only "true" if there are exactly {expected_count} matching elements, or "false" if the count is different.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and count the specified elements. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: count_elements_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for element counting: {response}")
                    return False
            else:
                print("⚠️ No AI response for element counting")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: count_elements_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


def min_elements_condition(selector: str, min_count: int, description: Optional[str] = None) -> Condition:
    """Check if there are at least N elements matching the selector using AI vision analysis"""
    if description is None:
        description = f"At least {min_count} elements with selector '{selector}'"
    
    def evaluator(context: GoalContext) -> bool:
        if not context.page_reference:
            return False
        
        try:
            # Take a screenshot of the current page
            screenshot = context.page_reference.screenshot(full_page=False)
            
            # Create a prompt for AI to count the elements
            prompt = f"""
Analyze the screenshot and count how many elements match the description.

Elements to count: {selector}
Minimum count required: {min_count}
Description: {description}

Instructions:
1. Look at the screenshot carefully
2. Count all elements that match the description
3. Consider that the elements might be:
   - Buttons, links, or interactive elements
   - Form fields, inputs, or controls
   - Images, icons, or visual elements
   - Any other UI elements that match the description
4. Count only visible elements on the current view

Return only "true" if there are at least {min_count} matching elements, or "false" if there are fewer.

Response format: Only return "true" or "false", nothing else.
"""
            
            # Use AI to analyze the screenshot
            response = generate_text(
                prompt=prompt,
                image=screenshot,
                system_prompt="You are a web page analyzer. Look at the screenshot and count the specified elements. Return only 'true' or 'false'.",
            )
            
            if response:
                response_clean = response.strip().lower()
                print(f"🔍 DEBUG: min_elements_condition - AI response: {response_clean}")
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for element counting: {response}")
                    return False
            else:
                print("⚠️ No AI response for element counting")
                return False
                
        except Exception as e:
            print(f"🔍 DEBUG: min_elements_condition - Exception: {e}")
            return False
    
    return create_environment_condition(description, evaluator)


# =============================================================================
# Computational Condition Evaluators
# =============================================================================

def is_weekday_condition(description: Optional[str] = None) -> Condition:
    """Check if today is a weekday (Monday-Friday)"""
    if description is None:
        description = "Today is a weekday"
    
    def evaluator(context: GoalContext) -> bool:
        today = datetime.datetime.now().weekday()
        return today < 5  # Monday=0, Sunday=6
    
    return create_computational_condition(description, evaluator)


def is_weekend_condition(description: Optional[str] = None) -> Condition:
    """Check if today is a weekend (Saturday-Sunday)"""
    if description is None:
        description = "Today is a weekend"
    
    def evaluator(context: GoalContext) -> bool:
        today = datetime.datetime.now().weekday()
        return today >= 5  # Saturday=5, Sunday=6
    
    return create_computational_condition(description, evaluator)


def is_specific_day_condition(day_name: str, description: Optional[str] = None) -> Condition:
    """Check if today is a specific day of the week"""
    if description is None:
        description = f"Today is {day_name}"
    
    def evaluator(context: GoalContext) -> bool:
        today = datetime.datetime.now().strftime('%A').lower()
        return today == day_name.lower()
    
    return create_computational_condition(description, evaluator)


def time_between_condition(start_hour: int, end_hour: int, description: Optional[str] = None) -> Condition:
    """Check if current time is between start_hour and end_hour (24-hour format)"""
    if description is None:
        description = f"Current time is between {start_hour}:00 and {end_hour}:00"
    
    def evaluator(context: GoalContext) -> bool:
        current_hour = datetime.datetime.now().hour
        return start_hour <= current_hour < end_hour
    
    return create_computational_condition(description, evaluator)


def mathematical_condition(expression: str, description: Optional[str] = None) -> Condition:
    """Evaluate a mathematical expression using AI"""
    if description is None:
        description = f"Mathematical expression '{expression}' is true"
    
    def evaluator(context: GoalContext) -> bool:
        try:
            
            # Create a prompt for the AI to evaluate the mathematical expression
            prompt = f"""
                            Evaluate the following mathematical expression and return only "true" or "false":

                            Expression: {expression}

                            Rules:
                            - Return "true" if the expression evaluates to a truthy value
                            - Return "false" if the expression evaluates to a falsy value or is invalid
                            - Handle basic arithmetic: +, -, *, /, **, %
                            - Handle comparisons: ==, !=, <, >, <=, >=
                            - Handle logical operations: and, or, not
                            - Handle parentheses for grouping
                            - Return "false" for any invalid or unsafe expressions

                            Response format: Only return "true" or "false", nothing else.
                        """
            
            # Call AI to evaluate the expression
            response = generate_text(
                prompt=prompt,
  # Simple mathematical evaluation
                system_prompt="You are a mathematical expression evaluator. Return only 'true' or 'false'.",
            )
            
            if response:
                # Clean the response and check for true/false
                response_clean = response.strip().lower()
                if response_clean == "true":
                    return True
                elif response_clean == "false":
                    return False
                else:
                    print(f"⚠️ Unexpected AI response for expression '{expression}': {response}")
                    return False
            else:
                print(f"⚠️ No AI response for expression '{expression}'")
                return False
                
        except Exception as e:
            print(f"⚠️ Error evaluating mathematical expression '{expression}': {e}")
            return False
    
    return create_computational_condition(description, evaluator)


# =============================================================================
# AI-Powered Condition Parser
# =============================================================================

class ConditionParser:
    """
    AI-powered parser that can interpret natural language conditions
    and convert them to appropriate Condition objects.
    """
    
    def __init__(self):
        # Available condition creation functions
        self.condition_functions = {
            'element_exists': element_exists_condition,
            'element_visible': element_visible_condition,
            'text_contains': text_contains_condition,
            'url_contains': url_contains_condition,
            'form_field_filled': form_field_filled_condition,
            'count_elements': count_elements_condition,
            'min_elements': min_elements_condition,
            'is_weekday': is_weekday_condition,
            'is_weekend': is_weekend_condition,
            'is_specific_day': is_specific_day_condition,
            'time_between': time_between_condition,
            'mathematical': mathematical_condition,
        }
    
    def parse_condition(self, condition_text: str) -> Optional[Condition]:
        """
        Parse a natural language condition into a Condition object using AI.
        
        Args:
            condition_text: Natural language description of the condition
            
        Returns:
            Condition object or None if parsing fails
        """
        try:
            
            # Create a prompt for the AI to analyze the condition
            prompt = f"""
Analyze the following condition text and determine what type of condition it is and what parameters to use.

Condition text: "{condition_text}"

Available condition types:
1. element_exists - Check if an element exists on the page
   Parameters: selector (string)
   Example: "element #submit-button exists"

2. element_visible - Check if an element is visible on the page
   Parameters: selector (string)
   Example: "element .form-field is visible"

3. text_contains - Check if the page contains specific text
   Parameters: text (string)
   Example: "page contains 'Welcome'"

4. url_contains - Check if the current URL contains a specific fragment
   Parameters: url_fragment (string)
   Example: "url contains 'dashboard'"

5. form_field_filled - Check if a form field has a value
   Parameters: selector (string)
   Example: "field #email is filled"

6. count_elements - Check if there are exactly N elements matching the selector
   Parameters: selector (string), expected_count (integer)
   Example: "exactly 3 elements with selector '.item'"

7. min_elements - Check if there are at least N elements matching the selector
   Parameters: selector (string), min_count (integer)
   Example: "at least 1 elements with selector '.error'"

8. is_weekday - Check if today is a weekday (Monday-Friday)
   Parameters: none
   Example: "today is a weekday"

9. is_weekend - Check if today is a weekend (Saturday-Sunday)
   Parameters: none
   Example: "today is a weekend"

10. is_specific_day - Check if today is a specific day of the week
    Parameters: day_name (string)
    Example: "today is monday"

11. time_between - Check if current time is between start_hour and end_hour
    Parameters: start_hour (integer), end_hour (integer)
    Example: "time is between 9 and 17"

12. mathematical - Evaluate a mathematical expression
    Parameters: expression (string)
    Example: "2 + 2 == 4"

Response format (JSON only):
{{
    "condition_type": "condition_type_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "description": "Human readable description of what this condition checks"
}}

If the condition cannot be parsed or doesn't match any available types, return:
{{
    "condition_type": null,
    "parameters": {{}},
    "description": "Could not parse condition"
}}
"""
            
            # Call AI to analyze the condition
            response = generate_text(
                prompt=prompt,
                system_prompt="You are a condition parser. Analyze natural language conditions and return JSON with the appropriate condition type and parameters.",
            )
            
            if not response:
                print(f"⚠️ No AI response for condition: '{condition_text}'")
                return None
            
            # Parse the JSON response
            import json
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON response for condition '{condition_text}': {response}")
                return None
            
            condition_type = result.get('condition_type')
            parameters = result.get('parameters', {})
            description = result.get('description', condition_text)
            
            if not condition_type or condition_type not in self.condition_functions:
                print(f"⚠️ Unknown condition type '{condition_type}' for: '{condition_text}'")
                return None
            
            # Create the condition using the appropriate function
            condition_func = self.condition_functions[condition_type]
            
            if condition_type == 'element_exists':
                return condition_func(parameters.get('selector', ''), description)
            elif condition_type == 'element_visible':
                return condition_func(parameters.get('selector', ''), description)
            elif condition_type == 'text_contains':
                return condition_func(parameters.get('text', ''), description)
            elif condition_type == 'url_contains':
                return condition_func(parameters.get('url_fragment', ''), description)
            elif condition_type == 'form_field_filled':
                return condition_func(parameters.get('selector', ''), description)
            elif condition_type == 'count_elements':
                return condition_func(parameters.get('selector', ''), parameters.get('expected_count', 0), description)
            elif condition_type == 'min_elements':
                return condition_func(parameters.get('selector', ''), parameters.get('min_count', 0), description)
            elif condition_type == 'is_weekday':
                return condition_func(description)
            elif condition_type == 'is_weekend':
                return condition_func(description)
            elif condition_type == 'is_specific_day':
                return condition_func(parameters.get('day_name', ''), description)
            elif condition_type == 'time_between':
                return condition_func(parameters.get('start_hour', 0), parameters.get('end_hour', 24), description)
            elif condition_type == 'mathematical':
                return condition_func(parameters.get('expression', ''), description)
            else:
                print(f"⚠️ Unhandled condition type: {condition_type}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error parsing condition '{condition_text}': {e}")
            return None
    
    def suggest_condition_types(self) -> List[str]:
        """
        Suggest additional condition types that might be useful.
        
        Returns:
            List of suggested condition type descriptions
        """
        return [
            "Network conditions: Check if internet connection is available, check response times",
            "File system conditions: Check if files exist, check file sizes, check modification dates",
            "Database conditions: Check if records exist, check data values, check connection status",
            "API conditions: Check if external APIs are responding, check response codes",
            "User input conditions: Check if user has provided input, check input validation",
            "State machine conditions: Check current state in a state machine, check transitions",
            "Resource conditions: Check memory usage, check CPU usage, check disk space",
            "Security conditions: Check authentication status, check permissions, check certificates",
            "Performance conditions: Check response times, check throughput, check error rates",
            "Business logic conditions: Check business rules, check workflow states, check approvals"
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_condition_from_text(condition_text: str) -> Optional[Condition]:
    """
    Convenience function to create a condition from natural language text.
    
    Args:
        condition_text: Natural language description of the condition
        
    Returns:
        Condition object or None if parsing fails
    """
    parser = ConditionParser()
    return parser.parse_condition(condition_text)


def get_available_conditions() -> Dict[str, List[str]]:
    """
    Get a list of available condition types and examples.
    
    Returns:
        Dictionary mapping condition types to example descriptions
    """
    return {
        "Environment State Conditions": [
            "element #submit-button exists",
            "element .form-field is visible", 
            "page contains 'Welcome'",
            "url contains 'dashboard'",
            "field #email is filled",
            "exactly 3 elements with selector '.item'",
            "at least 1 elements with selector '.error'"
        ],
        "Computational Conditions": [
            "today is a weekday",
            "today is a weekend", 
            "today is monday",
            "time is between 9 and 17",
            "2 + 2 == 4",
            "10 > 5"
        ]
    }
