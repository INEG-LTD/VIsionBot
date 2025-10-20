"""
Click Goal - Monitors click interactions to verify correct elements are clicked.
"""
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ai_utils import generate_model
from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext, InteractionType
from .element_analyzer import ElementAnalyzer


class ElementMatchEvaluation(BaseModel):
    """Result of evaluating if a clicked element matches the target description"""
    is_match: bool = Field(description="Whether the clicked element matches the target")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the match assessment")
    reasoning: str = Field(description="Explanation of why this match determination was made")


class ClickGoal(BaseGoal):
    """
    Goal for clicking a specific element.
    
    This goal monitors click interactions and determines if the correct
    element was clicked based on the user's description.
    """
    
    # ClickGoal should be evaluated BEFORE interaction since clicking might navigate away
    EVALUATION_TIMING = EvaluationTiming.BEFORE
    
    def __init__(self, description: str, target_description: str, **kwargs):
        super().__init__(description, **kwargs)
        self.target_description = target_description.lower().strip()
        self.planned_click_coordinates: Optional[Tuple[int, int]] = None
        self.planned_element_info: Optional[Dict[str, Any]] = None
        self.actual_click_coordinates: Optional[Tuple[int, int]] = None
        self.actual_element_info: Optional[Dict[str, Any]] = None
        self.click_attempted = False
        self.click_successful = False
        self.element_analyzer: Optional[ElementAnalyzer] = None
        
        # Track failed attempts to avoid retrying the same coordinates/elements
        self.failed_coordinates: List[Tuple[int, int]] = []
        self.failed_element_descriptions: List[str] = []
        self.failed_reasons: List[str] = []
    
    def on_monitoring_start(self) -> None:
        """Initialize element analyzer when monitoring starts"""
        # Element analyzer will be set by the goal monitor when page is available
        pass
    
    def set_element_analyzer(self, analyzer: ElementAnalyzer) -> None:
        """Set the element analyzer (called by goal monitor)"""
        self.element_analyzer = analyzer
    
    def record_failed_attempt(self, coordinates: Tuple[int, int], element_description: str, reason: str) -> None:
        """Record a failed attempt to avoid retrying the same coordinates/elements"""
        if coordinates not in self.failed_coordinates:
            self.failed_coordinates.append(coordinates)
        if element_description and element_description not in self.failed_element_descriptions:
            self.failed_element_descriptions.append(element_description)
        if reason not in self.failed_reasons:
            self.failed_reasons.append(reason)
        print(f"[ClickGoal] Recorded failed attempt: coords={coordinates}, element='{element_description}', reason='{reason}'")
    
    def get_retry_context(self) -> str:
        """Get context about failed attempts for retry planning"""
        if not self.failed_coordinates and not self.failed_element_descriptions and not self.failed_reasons:
            return ""
        
        context_parts = []
        
        if self.failed_coordinates:
            coords_str = ", ".join([f"({x}, {y})" for x, y in self.failed_coordinates])
            context_parts.append(f"Previously failed coordinates: {coords_str}")
        
        if self.failed_element_descriptions:
            elements_str = ", ".join([f"'{desc}'" for desc in self.failed_element_descriptions])
            context_parts.append(f"Previously failed elements: {elements_str}")
        
        if self.failed_reasons:
            reasons_str = ", ".join([f"'{reason}'" for reason in self.failed_reasons])
            context_parts.append(f"Previous failure reasons: {reasons_str}")
        
        return " | ".join(context_parts)
    
    def record_planned_click(self, x: int, y: int, element_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the coordinates and element info for a planned click.
        This should be called BEFORE the click happens.
        """
        self.planned_click_coordinates = (x, y)
        self.planned_element_info = element_info
        print(f"[ClickGoal] Recorded planned click at ({x}, {y})")
    
    def on_interaction(self, interaction) -> None:
        """Track click interactions"""
        if interaction.interaction_type == InteractionType.CLICK:
            self.click_attempted = True
            self.actual_click_coordinates = interaction.coordinates
            self.actual_element_info = interaction.target_element_info
            self.click_successful = interaction.success
            print(f"[ClickGoal] Recorded actual click at {interaction.coordinates}")
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate if the correct element was/will be clicked.
        
        For ClickGoal with BEFORE timing, this evaluates the planned click.
        For post-interaction evaluation, this evaluates the completed click.
        """
        # Check if we have planned interaction data (pre-interaction evaluation)
        planned_interaction = getattr(context, 'planned_interaction', None)
        if planned_interaction and planned_interaction.get('interaction_type') == InteractionType.CLICK:
            return self._evaluate_planned_click(context, planned_interaction)
        
        # Fallback to post-interaction evaluation
        if not self.click_attempted:
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No click interaction has been attempted yet",
                next_actions=["Wait for click interaction"]
            )
        
        if not self.click_successful:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.9,
                reasoning="Click interaction failed to execute",
                evidence={"error": "Click execution failed"}
            )
        
        # Lazy loading: Only analyze element when actually needed for evaluation
        if not self.actual_element_info and self.actual_click_coordinates and self.element_analyzer:
            x, y = self.actual_click_coordinates
            print(f"[ClickGoal] Lazy loading element analysis at ({x}, {y})")
            self.actual_element_info = self.element_analyzer.analyze_element_at_coordinates(x, y)
        
        if not self.actual_element_info:
            # Record this failed attempt
            self.record_failed_attempt(self.actual_click_coordinates, "", "Could not analyze clicked element")
            
            # Request retry if we haven't exceeded max retries
            if self.can_retry():
                retry_reason = f"Could not analyze clicked element at {self.actual_click_coordinates} - element analysis failed"
                if self.request_retry(retry_reason):
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.3,
                        reasoning="Could not analyze the clicked element, requesting retry",
                        evidence={
                            "coordinates": self.actual_click_coordinates,
                            "retry_requested": True,
                            "retry_count": self.retry_count,
                            "retry_context": self.get_retry_context()
                        },
                        next_actions=["Retry plan generation to find valid clickable element"]
                    )
            
            # Max retries exceeded, fail the goal
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.3,
                reasoning="Could not analyze the clicked element and max retries exceeded",
                evidence={
                    "coordinates": self.actual_click_coordinates,
                    "retry_count": self.retry_count,
                    "max_retries_exceeded": True
                }
            )
        
        # Check if the clicked element was actually clickable
        is_clickable = self.actual_element_info.get('isClickable', False)
        if not is_clickable:
            # Get element description for tracking
            element_description = ""
            try:
                if self.element_analyzer and context.current_state.screenshot:
                    element_description = self.element_analyzer.get_element_description_with_ai(
                        self.actual_element_info,
                        context.current_state.screenshot,
                        *self.actual_click_coordinates
                    )
                else:
                    element_description = self._create_fallback_description(self.actual_element_info)
            except Exception:
                element_description = self._create_fallback_description(self.actual_element_info)
            
            # Record this failed attempt
            self.record_failed_attempt(self.actual_click_coordinates, element_description, "Clicked element was not clickable")
            
            # Request retry if we haven't exceeded max retries
            if self.can_retry():
                retry_reason = f"Clicked element at {self.actual_click_coordinates} was not clickable - need to find clickable element"
                if self.request_retry(retry_reason):
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.9,
                        reasoning=f"Clicked element at {self.actual_click_coordinates} was not clickable, requesting retry",
                        evidence={
                            "coordinates": self.actual_click_coordinates,
                            "element_info": self.actual_element_info,
                            "is_clickable": is_clickable,
                            "retry_requested": True,
                            "retry_count": self.retry_count,
                            "retry_context": self.get_retry_context()
                        },
                        next_actions=["Retry plan generation to find clickable element"]
                    )
            
            # Max retries exceeded, fail the goal
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.9,
                reasoning=f"Clicked element at {self.actual_click_coordinates} was not clickable and max retries exceeded",
                evidence={
                    "coordinates": self.actual_click_coordinates,
                    "element_info": self.actual_element_info,
                    "is_clickable": is_clickable,
                    "retry_count": self.retry_count,
                    "max_retries_exceeded": True
                }
            )
        
        # Lazy loading: Get AI description only when needed for evaluation
        try:
            if self.element_analyzer and context.current_state.screenshot:
                print(f"[ClickGoal] Lazy loading AI description for element at {self.actual_click_coordinates}")
                element_description = self.element_analyzer.get_element_description_with_ai(
                    self.actual_element_info,
                    context.current_state.screenshot,
                    *self.actual_click_coordinates
                )
            else:
                # Fallback description
                element_description = self._create_fallback_description(self.actual_element_info)
            
            # Compare with target using vision-assisted evaluation
            page_dims = (
                context.current_state.page_width,
                context.current_state.page_height,
            ) if context.current_state else None
            match_result = self._evaluate_element_match(
                element_description,
                self.target_description,
                self.actual_element_info,
                page_dims,
            )
            
            if match_result["is_match"]:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=match_result["confidence"],
                    reasoning=f"Successfully clicked '{element_description}' which matches target '{self.target_description}'",
                    evidence={
                        "clicked_element": element_description,
                        "target_element": self.target_description,
                        "coordinates": self.actual_click_coordinates,
                        "element_info": self.actual_element_info,
                        "match_reasoning": match_result["reasoning"]
                    }
                )
            else:
                # Record this failed attempt
                self.record_failed_attempt(self.actual_click_coordinates, element_description, f"Wrong element clicked: '{element_description}' instead of '{self.target_description}'")
                
                # Request retry if we haven't exceeded max retries
                if self.can_retry():
                    retry_reason = f"Wrong element clicked: '{element_description}' instead of '{self.target_description}'"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=match_result["confidence"],
                            reasoning=f"Wrong element clicked, requesting retry: {match_result['reasoning']}",
                            evidence={
                                "clicked_element": element_description,
                                "target_element": self.target_description,
                                "coordinates": self.actual_click_coordinates,
                                "element_info": self.actual_element_info,
                                "match_reasoning": match_result["reasoning"],
                                "retry_requested": True,
                                "retry_count": self.retry_count,
                                "retry_context": self.get_retry_context()
                            },
                            next_actions=["Retry plan generation to find correct element"]
                        )
                
                # Max retries exceeded, fail the goal
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=match_result["confidence"],
                    reasoning=f"Clicked '{element_description}' but target was '{self.target_description}' - {match_result['reasoning']}. Max retries exceeded.",
                    evidence={
                        "clicked_element": element_description,
                        "target_element": self.target_description,
                        "coordinates": self.actual_click_coordinates,
                        "element_info": self.actual_element_info,
                        "match_reasoning": match_result["reasoning"],
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
        except Exception as e:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.2,
                reasoning=f"Error evaluating click goal: {e}",
                evidence={"error": str(e)}
            )
    
    def _evaluate_planned_click(self, context: GoalContext, planned_interaction: Dict[str, Any]) -> GoalResult:
        """
        Evaluate a planned click interaction before it happens.
        """
        coordinates = planned_interaction.get('coordinates')
        if not coordinates:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No coordinates provided for planned click"
            )
        
        x, y = coordinates
        print(f"[ClickGoal] Evaluating planned click at ({x}, {y})")
        
        # Analyze the element that's about to be clicked
        if not self.element_analyzer:
            print("[ClickGoal] No element analyzer available for pre-click evaluation")
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No element analyzer available for pre-click evaluation"
            )
        
        try:
            # Lazy loading: Only analyze element when needed for evaluation
            element_info = None
            
            # If a specific target selector was provided, prefer analyzing that element
            selector = planned_interaction.get('target_selector')
            if selector and self.element_analyzer:
                print(f"[ClickGoal] Lazy loading element analysis by selector: {selector}")
                element_info = self.element_analyzer.analyze_element_by_selector(selector)
                if element_info:
                    print(f"[ClickGoal] Element info (by selector): {element_info}")
                    if element_info.get('isClickable', False):
                        # Describe and match target
                        screenshot = planned_interaction.get('screenshot')
                        if screenshot:
                            print("[ClickGoal] Getting element description with AI")
                            element_description = self.element_analyzer.get_element_description_with_ai(
                                element_info, screenshot, x, y
                            )
                        else:
                            element_description = self._create_fallback_description(element_info)
                        print(f"[ClickGoal] About to click (by selector): '{element_description}'")
                        page_dims = (
                            context.current_state.page_width,
                            context.current_state.page_height,
                        ) if context.current_state else None
                        match_result = self._evaluate_element_match(
                            element_description,
                            self.target_description,
                            element_info,
                            page_dims,
                        )
                        if match_result["is_match"]:
                            return GoalResult(
                                status=GoalStatus.ACHIEVED,
                                confidence=match_result["confidence"],
                                reasoning=f"About to click '{element_description}' which matches target '{self.target_description}'",
                                evidence={
                                    "target_element": element_description,
                                    "expected_target": self.target_description,
                                    "coordinates": (x, y),
                                    "element_info": element_info,
                                    "match_reasoning": match_result["reasoning"],
                                    "evaluation_timing": "pre_interaction",
                                    "target_selector": selector,
                                }
                            )
                    # If not clickable by selector, fall back to coordinate analysis below
            
            # Lazy loading: Only analyze by coordinates if selector analysis failed or wasn't available
            if not element_info:
                print(f"[ClickGoal] Lazy loading element analysis at coordinates ({x}, {y})")
                element_info = self.element_analyzer.analyze_element_at_coordinates(x, y)
                print(f"[ClickGoal] Element info: {element_info}")
            
            if not element_info or element_info.get("error"):
                print(f"[ClickGoal] Could not analyze element at ({x}, {y}) before click")
                
                # Record this failed attempt
                error_msg = element_info.get("error") if element_info else "No element found"
                self.record_failed_attempt((x, y), "", f"Could not analyze element: {error_msg}")
                
                # Request retry if we haven't exceeded max retries
                if self.can_retry():
                    retry_reason = f"Could not analyze element at ({x}, {y}) - coordinates may be invalid or element not found"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.2,
                            reasoning=f"Could not analyze element at ({x}, {y}), requesting retry to find valid element",
                            evidence={
                                "coordinates": (x, y),
                                "error": error_msg,
                                "retry_requested": True,
                                "retry_count": self.retry_count,
                                "retry_context": self.get_retry_context()
                            },
                            next_actions=["Retry plan generation to find valid clickable element"]
                        )
                
                # Max retries exceeded, fail the goal
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.2,
                    reasoning=f"Could not analyze element at ({x}, {y}) and max retries exceeded",
                    evidence={
                        "coordinates": (x, y),
                        "error": element_info.get("error") if element_info else "No element found",
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
            # Check if element is clickable
            is_clickable = element_info.get('isClickable', False)
            if not is_clickable:
                print(f"[ClickGoal] Element at ({x}, {y}) is not clickable")
                
                # Get element description for tracking
                element_description = ""
                screenshot = planned_interaction.get('screenshot')
                if screenshot:
                    try:
                        element_description = self.element_analyzer.get_element_description_with_ai(
                            element_info, screenshot, x, y
                        )
                    except Exception:
                        element_description = self._create_fallback_description(element_info)
                else:
                    element_description = self._create_fallback_description(element_info)
                
                # Record this failed attempt
                self.record_failed_attempt((x, y), element_description, "Element is not clickable")
                
                # Request retry if we haven't exceeded max retries
                if self.can_retry():
                    retry_reason = f"Element at ({x}, {y}) is not clickable - need to find a clickable element"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.9,
                            reasoning=f"Element at ({x}, {y}) is not clickable, requesting retry to find clickable element",
                            evidence={
                                "coordinates": (x, y),
                                "element_info": element_info,
                                "is_clickable": is_clickable,
                                "retry_requested": True,
                                "retry_count": self.retry_count,
                                "retry_context": self.get_retry_context()
                            },
                            next_actions=["Retry plan generation to find clickable element"]
                        )
                
                # Max retries exceeded, fail the goal
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.9,
                    reasoning=f"Element at ({x}, {y}) is not clickable and max retries exceeded",
                    evidence={
                        "coordinates": (x, y),
                        "element_info": element_info,
                        "is_clickable": is_clickable,
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
            # Get AI description of the element about to be clicked
            screenshot = planned_interaction.get('screenshot')
            if screenshot:
                print("[ClickGoal] Getting element description with AI")
                element_description = self.element_analyzer.get_element_description_with_ai(
                    element_info, screenshot, x, y
                )
            else:
                print("[ClickGoal] No screenshot available for pre-click evaluation")
                element_description = self._create_fallback_description(element_info)
            
            print(f"[ClickGoal] About to click: '{element_description}'")
            
            # Compare with target using AI
            page_dims = (
                context.current_state.page_width,
                context.current_state.page_height,
            ) if context.current_state else None
            match_result = self._evaluate_element_match(
                element_description,
                self.target_description,
                element_info,
                page_dims,
            )
            
            if match_result["is_match"]:
                print(f"[ClickGoal] About to click: '{element_description}' which matches target '{self.target_description}'")
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=match_result["confidence"],
                    reasoning=f"About to click '{element_description}' which matches target '{self.target_description}' - {match_result['reasoning']}",
                    evidence={
                        "target_element": element_description,
                        "expected_target": self.target_description,
                        "coordinates": (x, y),
                        "element_info": element_info,
                        "match_reasoning": match_result["reasoning"],
                        "evaluation_timing": "pre_interaction"
                    }
                )
            else:
                print(f"[ClickGoal] About to click: '{element_description}' but target was '{self.target_description}'")
                
                # Record this failed attempt
                self.record_failed_attempt((x, y), element_description, f"Wrong element: '{element_description}' instead of '{self.target_description}'")
                
                # Request retry if we haven't exceeded max retries
                if self.can_retry():
                    retry_reason = f"Wrong element detected: '{element_description}' instead of '{self.target_description}'"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=match_result["confidence"],
                            reasoning=f"Wrong element detected, requesting retry: {match_result['reasoning']}",
                            evidence={
                                "target_element": element_description,
                                "expected_target": self.target_description,
                                "coordinates": (x, y),
                                "element_info": element_info,
                                "match_reasoning": match_result["reasoning"],
                                "evaluation_timing": "pre_interaction",
                                "retry_requested": True,
                                "retry_count": self.retry_count,
                                "retry_context": self.get_retry_context()
                            },
                            next_actions=["Retry plan generation to find correct element"]
                        )
                
                # Max retries exceeded, fail the goal
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=match_result["confidence"],
                    reasoning=f"About to click '{element_description}' but target was '{self.target_description}' - {match_result['reasoning']}. Max retries exceeded.",
                    evidence={
                        "target_element": element_description,
                        "expected_target": self.target_description,
                        "coordinates": (x, y),
                        "element_info": element_info,
                        "match_reasoning": match_result["reasoning"],
                        "evaluation_timing": "pre_interaction",
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
        except Exception as e:
            print(f"[ClickGoal] Error in pre-click evaluation: {e}")
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning=f"Error in pre-click evaluation: {e}",
                evidence={"error": str(e), "coordinates": (x, y)}
            )
    
    def _create_fallback_description(self, element_info: Dict[str, Any]) -> str:
        """Create a basic description when AI is not available"""
        element_type = element_info.get('elementType', 'element')
        text = element_info.get('text', '')[:50]
        print(f"[GoalFramework] Element description with fallback: {element_type} containing '{text}'")
        if text:
            return f"{element_type} containing '{text}'"
        else:
            print(f"[GoalFramework] Element description with fallback: {element_type}")
            return element_type
    
    def _evaluate_element_match(
        self,
        actual_description: str,
        target_description: str,
        actual_element_info: Optional[Dict[str, Any]] = None,
        page_dimensions: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Use single AI evaluation method for element matching.
        Simplified to avoid redundant vision scoring and fallback matching.
        """
        try:
            print(f"[GoalFramework] Evaluating element match: {actual_description} vs {target_description}")
            
            # Enhanced system prompt with element context
            element_context = ""
            if actual_element_info:
                element_context = f"""
                Element details:
                - Type: {actual_element_info.get('elementType', 'unknown')}
                - Text: {actual_element_info.get('text', '')[:100]}
                - Clickable: {actual_element_info.get('isClickable', False)}
                - Attributes: {str(actual_element_info.get('attributes', {}))[:200]}
                """
            
            system_prompt = f"""
            You are evaluating whether a clicked UI element matches the user's intent.

            User wanted to click: "{target_description}"
            Actually clicked: "{actual_description}"
            {element_context}

            Determine if these match semantically. Consider:
            - Synonyms and similar terms (e.g., "button" vs "btn", "link" vs "anchor")
            - Partial matches (e.g., "first link" could match "navigation link")
            - Context and intent (e.g., "submit" could match "submit button" or "send button")
            - Element functionality and purpose

            Provide your evaluation with confidence and reasoning.
            """

            result = generate_model(
                prompt="Evaluate if the clicked element matches the target description.",
                model_object_type=ElementMatchEvaluation,
                system_prompt=system_prompt,
            )

            return result.model_dump()

        except Exception as e:
            print(f"AI evaluation failed: {e}, falling back to simple matching")
            return self._simple_string_match(actual_description, target_description)
    
    def _simple_string_match(self, actual: str, target: str) -> Dict[str, Any]:
        """Simple fallback matching when AI is not available"""
        actual_lower = actual.lower()
        target_lower = target.lower()
        
        # Exact match
        if actual_lower == target_lower:
            return {"is_match": True, "confidence": 1.0, "reasoning": "Exact match"}
        
        # Substring match
        if target_lower in actual_lower or actual_lower in target_lower:
            return {"is_match": True, "confidence": 0.8, "reasoning": "Substring match"}
        
        # Word overlap
        actual_words = set(actual_lower.split())
        target_words = set(target_lower.split())
        overlap = actual_words & target_words
        
        if overlap and len(overlap) >= len(target_words) * 0.5:
            return {
                "is_match": True, 
                "confidence": 0.6, 
                "reasoning": f"Word overlap: {overlap}"
            }
        
        return {"is_match": False, "confidence": 0.9, "reasoning": "No significant match found"}
    
    def get_description(self, context: GoalContext) -> str:
        """
        Generate a detailed description of what this click goal is looking for.
        
        The description should include:
        - Goal statement and target description
        - Current click status (attempted, successful, failed)
        - Planned vs actual click information
        - Element descriptions for context
        
        Format should be:
        ```
        Click goal: [description]
        Target: [target_description]
        Status: [current_status]
        Actually clicked: [element_description] (if attempted)
        Planned click coordinates: (x, y) (if planned)
        Planned to click: [element_description] (if planned)
        ```
        """
        description_parts = []
        
        # Main goal description
        description_parts.append(f"Click goal: {self.description}")
        description_parts.append(f"Target: {self.target_description}")
        
        # Add status information if available
        if self.click_attempted:
            if self.click_successful:
                description_parts.append("Status: Click has been attempted and executed successfully")
                if self.actual_element_info:
                    element_desc = self._create_fallback_description(self.actual_element_info)
                    description_parts.append(f"Actually clicked: {element_desc}")
            else:
                description_parts.append("Status: Click was attempted but failed to execute")
        else:
            description_parts.append("Status: No click interaction attempted yet")
        
        # Add planned click info if available
        if self.planned_click_coordinates:
            x, y = self.planned_click_coordinates
            description_parts.append(f"Planned click coordinates: ({x}, {y})")
            if self.planned_element_info:
                element_desc = self._create_fallback_description(self.planned_element_info)
                description_parts.append(f"Planned to click: {element_desc}")
        
        # Add retry context if available
        retry_context = self.get_retry_context()
        if retry_context:
            description_parts.append(f"Retry context: {retry_context}")
        
        return "\n".join(description_parts)
