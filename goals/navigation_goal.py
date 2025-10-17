"""
Navigation Goal - Analyzes navigation targets by previewing pages in new tabs.
"""
import time
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from ai_utils import generate_model
from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext, InteractionType


class NavigationAnalysis(BaseModel):
    """Result of AI analysis of a navigation target page"""
    matches_intent: bool = Field(description="Whether the page matches the navigation intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis")
    reasoning: str = Field(description="Explanation of the analysis")
    page_summary: str = Field(description="Brief summary of what the page contains")


class NavigationGoal(BaseGoal):
    """
    Goal for page navigation tasks using preview analysis.
    
    This goal analyzes navigation targets by:
    1. Opening the target URL in a new tab (before clicking)
    2. Taking full-page screenshots
    3. Using AI to analyze if it matches the navigation intent
    4. Closing the preview tab
    5. Proceeding with navigation if it's the right destination
    """
    
    # NavigationGoal uses BOTH timing - BEFORE for direct URLs, AFTER for button results
    EVALUATION_TIMING = EvaluationTiming.BOTH
    
    def __init__(
        self, 
        description: str, 
        navigation_intent: str,
        target_url_contains: List[str] = None, 
        target_page_text: List[str] = None, 
        **kwargs
    ):
        super().__init__(description, **kwargs)
        self.navigation_intent = navigation_intent  # What the user is looking for
        self.target_url_contains = target_url_contains or []
        self.target_page_text = target_page_text or []
        self.preview_results: Optional[Dict[str, Any]] = None
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate navigation goal using different strategies based on timing.
        
        BEFORE interaction: Only Strategy 1 (direct URL preview)
        AFTER interaction: Strategies 2 & 3 (Ctrl+Click and current page analysis)
        """
        # Check if we have planned interaction data (pre-interaction evaluation)
        planned_interaction = getattr(context, 'planned_interaction', None)
        if planned_interaction and planned_interaction.get('interaction_type') == InteractionType.CLICK:
            return self._evaluate_before_click(context, planned_interaction)
        
        # Post-interaction evaluation: check if navigation actually happened
        return self._evaluate_after_click(context)
    
    def _evaluate_before_click(self, context: GoalContext, planned_interaction: Dict[str, Any]) -> GoalResult:
        """
        BEFORE interaction: Only try Strategy 1 (direct URL preview).
        If no direct URL is available, return PENDING to wait for post-interaction evaluation.
        """
        coordinates = planned_interaction.get('coordinates')
        if not coordinates:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No coordinates provided for navigation preview"
            )
        
        x, y = coordinates
        print(f"[NavigationGoal] Pre-click evaluation at ({x}, {y})")
        
        # Strategy 1: Try direct URL preview only
        target_url = self._extract_target_url_from_coordinates(context, x, y)
        
        if target_url:
            print(f"[NavigationGoal] Found direct URL for pre-click analysis: {target_url}")
            preview_analysis = self._preview_page_in_new_tab(target_url, context)
            
            if preview_analysis:
                self.preview_results = preview_analysis
                
                if preview_analysis["matches_intent"]:
                    return GoalResult(
                        status=GoalStatus.ACHIEVED,
                        confidence=preview_analysis["confidence"],
                        reasoning=f"Pre-click URL preview confirms target matches intent: {preview_analysis['reasoning']}",
                        evidence={
                            "target_url": target_url,
                            "navigation_intent": self.navigation_intent,
                            "page_summary": preview_analysis["page_summary"],
                            "preview_analysis": preview_analysis,
                            "evaluation_timing": "pre_interaction_url_preview",
                            "strategy": "direct_url_preview"
                        }
                    )
                else:
                    # Request retry if we haven't exceeded max retries
                    if self.can_retry():
                        retry_reason = f"Navigation target doesn't match intent: {preview_analysis['reasoning']}"
                        if self.request_retry(retry_reason):
                            return GoalResult(
                                status=GoalStatus.PENDING,
                                confidence=preview_analysis["confidence"],
                                reasoning=f"Navigation target doesn't match intent, requesting retry: {preview_analysis['reasoning']}",
                                evidence={
                                    "target_url": target_url,
                                    "navigation_intent": self.navigation_intent,
                                    "page_summary": preview_analysis["page_summary"],
                                    "preview_analysis": preview_analysis,
                                    "evaluation_timing": "pre_interaction_url_preview",
                                    "strategy": "direct_url_preview",
                                    "retry_requested": True,
                                    "retry_count": self.retry_count
                                },
                                next_actions=["Retry plan generation to find correct navigation target"]
                            )
                    
                    # Max retries exceeded, fail the goal
                    return GoalResult(
                        status=GoalStatus.FAILED,
                        confidence=preview_analysis["confidence"],
                        reasoning=f"Pre-click URL preview shows target does not match intent: {preview_analysis['reasoning']}. Max retries exceeded.",
                        evidence={
                            "target_url": target_url,
                            "navigation_intent": self.navigation_intent,
                            "page_summary": preview_analysis["page_summary"],
                            "preview_analysis": preview_analysis,
                            "evaluation_timing": "pre_interaction_url_preview",
                            "strategy": "direct_url_preview",
                            "retry_count": self.retry_count,
                            "max_retries_exceeded": True
                        }
                    )
        
        # No direct URL available - wait for post-interaction evaluation
        print("[NavigationGoal] No direct URL found, will evaluate after interaction")
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=1.0,
            reasoning="No direct URL available for pre-click preview, waiting for post-interaction evaluation",
            evidence={"strategy": "waiting_for_post_interaction"}
        )
    
    def _evaluate_after_click(self, context: GoalContext) -> GoalResult:
        """
        AFTER interaction: Use strategies 2 & 3 or check if navigation actually occurred.
        """
        print("[NavigationGoal] Post-click evaluation")
        
        # Check if we already have a successful pre-click evaluation
        if self.preview_results and self.preview_results.get("matches_intent"):
            print("[NavigationGoal] Using previous pre-click analysis results")
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=self.preview_results["confidence"],
                reasoning=f"Navigation completed successfully based on pre-click analysis: {self.preview_results['reasoning']}",
                evidence={
                    "navigation_intent": self.navigation_intent,
                    "preview_analysis": self.preview_results,
                    "evaluation_timing": "post_interaction_confirmation",
                    "strategy": "pre_click_confirmed"
                }
            )
        
        # Check if URL actually changed (successful navigation)
        if len(context.url_history) >= 2:
            previous_url = context.url_history[-2]
            current_url = context.current_state.url
            
            if previous_url != current_url:
                print(f"[NavigationGoal] Navigation detected: {previous_url} -> {current_url}")
                
                # Analyze the new page
                analysis = self._analyze_current_page_as_target(context)
                if analysis:
                    analysis["reasoning"] = f"Post-navigation analysis: {analysis['reasoning']}"
                    analysis["page_summary"] = f"Navigated to: {analysis['page_summary']}"
                    
                    return GoalResult(
                        status=GoalStatus.ACHIEVED if analysis["matches_intent"] else GoalStatus.FAILED,
                        confidence=analysis["confidence"],
                        reasoning=analysis["reasoning"],
                        evidence={
                            "previous_url": previous_url,
                            "current_url": current_url,
                            "navigation_intent": self.navigation_intent,
                            "page_analysis": analysis,
                            "evaluation_timing": "post_interaction_navigation",
                            "strategy": "post_navigation_analysis"
                        }
                    )
        
        # No navigation detected - analyze current page to see if it matches intent
        print("[NavigationGoal] No navigation detected, analyzing current page")
        analysis = self._analyze_current_page_as_target(context)
        if analysis:
            return GoalResult(
                status=GoalStatus.ACHIEVED if analysis["matches_intent"] else GoalStatus.PENDING,
                confidence=min(0.6, analysis["confidence"]),  # Lower confidence since no navigation
                reasoning=f"No navigation occurred, current page analysis: {analysis['reasoning']}",
                evidence={
                    "navigation_intent": self.navigation_intent,
                    "page_analysis": analysis,
                    "evaluation_timing": "post_interaction_no_navigation",
                    "strategy": "current_page_analysis"
                }
            )
        
        return GoalResult(
            status=GoalStatus.UNKNOWN,
            confidence=0.2,
            reasoning="Could not evaluate navigation goal after interaction"
        )
    

    
    def _extract_target_url_from_coordinates(self, context: GoalContext, x: int, y: int) -> Optional[str]:
        """
        Extract the target URL from the element at the given coordinates.
        """
        try:
            page = context.page_reference
            if not page:
                print("[NavigationGoal] No page reference available in context")
                return None
            
            # Extract href from element at coordinates
            target_url = page.evaluate("""
            (coords) => {
                const x = coords.x;
                const y = coords.y;
                const element = document.elementFromPoint(x, y);
                if (!element) return null;
                
                // Look for href in the element or its parents
                let current = element;
                while (current && current !== document.body) {
                    if (current.tagName === 'A' && current.href) {
                        return current.href;
                    }
                    if (current.onclick) {
                        // Try to extract URL from onclick if it's a navigation
                        const onclickStr = current.onclick.toString();
                        const urlMatch = onclickStr.match(/(?:location\.href|window\.open|navigate).*?['"`]([^'"`]+)['"`]/);
                        if (urlMatch) return urlMatch[1];
                    }
                    current = current.parentElement;
                }
                
                return null;
            }
            """, {"x": x, "y": y})
            
            if target_url:
                # Resolve relative URLs to absolute
                if target_url.startswith('/'):
                    base_url = page.url
                    from urllib.parse import urljoin
                    target_url = urljoin(base_url, target_url)
                
                print(f"[NavigationGoal] Extracted target URL: {target_url}")
                return target_url
            else:
                print(f"[NavigationGoal] No URL found at coordinates ({x}, {y})")
                return None
            
        except Exception as e:
            print(f"[NavigationGoal] Error extracting target URL: {e}")
            return None
    
    def _preview_page_in_new_tab(self, target_url: str, context: GoalContext) -> Optional[Dict[str, Any]]:
        """
        Open the target URL in a new tab, analyze it with AI, then close the tab.
        """
        preview_tab = None
        try:
            page = context.page_reference
            if not page:
                print("[NavigationGoal] No page reference available for preview")
                return None
            
            print(f"[NavigationGoal] Opening preview tab for: {target_url}")
            
            # Get browser context and create new tab
            browser_context = page.context
            preview_tab = browser_context.new_page()
            
            # Set a reasonable viewport for consistent screenshots
            preview_tab.set_viewport_size({"width": 1280, "height": 800})
            
            # Navigate to target URL with timeout
            print(f"[NavigationGoal] Loading preview page: {target_url}")
            preview_tab.goto(target_url, wait_until="domcontentloaded", timeout=10000)
            
            # Wait a moment for dynamic content to load
            time.sleep(2)
            
            # Analyze the loaded page
            analysis = self._analyze_tab_page(preview_tab, target_url)
            
            return analysis
            
        except Exception as e:
            print(f"[NavigationGoal] Error in page preview: {e}")
            return None
        finally:
            # Always close the preview tab
            if preview_tab:
                try:
                    print("[NavigationGoal] Closing preview tab")
                    preview_tab.close()
                except Exception as e:
                    print(f"[NavigationGoal] Error closing preview tab: {e}")
    
    def _analyze_tab_page(self, tab_page, target_url: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a page in a tab (either preview tab or Ctrl+Click tab).
        This is the common analysis function used by different preview methods.
        """
        try:
            # Take full-page screenshot for AI analysis
            print("[NavigationGoal] Taking full-page screenshot for analysis")
            screenshot = tab_page.screenshot(full_page=True)
            
            # Get page title and visible text for additional context
            page_title = tab_page.title()
            visible_text = tab_page.evaluate("document.body.innerText")[:2000]
            
            print(f"[NavigationGoal] Analyzing page: {page_title} at {target_url}")
            
            # Analyze with AI
            analysis = self._analyze_page_with_ai(
                screenshot=screenshot,
                target_url=target_url,
                navigation_intent=self.navigation_intent,
                page_title=page_title,
                visible_text=visible_text
            )
            
            return analysis
            
        except Exception as e:
            print(f"[NavigationGoal] Error analyzing tab page: {e}")
            return None
    
    def _analyze_current_page_as_target(self, context: GoalContext) -> Optional[Dict[str, Any]]:
        """
        Analyze the current page directly as the navigation target.
        This is used when we can't preview in a new tab or use Ctrl+Click.
        """
        page = context.page_reference
        if not page:
            return None
        
        try:
            print("[NavigationGoal] Analyzing current page as navigation target")
            
            # Get current page information
            current_url = page.url
            current_title = page.title()
            
            # Take full-page screenshot of current page
            screenshot = page.screenshot(full_page=True)
            visible_text = page.evaluate("document.body.innerText")[:2000]
            
            print(f"[NavigationGoal] Current page: {current_title} at {current_url}")
            
            # Analyze current page as if it's the target
            analysis = self._analyze_page_with_ai(
                screenshot=screenshot,
                target_url=current_url,
                navigation_intent=self.navigation_intent,
                page_title=current_title,
                visible_text=visible_text
            )
            
            if analysis:
                analysis["reasoning"] = f"Current page analysis (no preview possible): {analysis['reasoning']}"
                analysis["page_summary"] = f"Current page: {analysis['page_summary']}"
                # Slightly lower confidence since we're not previewing the actual destination
                analysis["confidence"] = min(0.8, analysis.get("confidence", 0.5))
            
            return analysis
            
        except Exception as e:
            print(f"[NavigationGoal] Error analyzing current page: {e}")
            return None
    
    def _analyze_page_with_ai(
        self, 
        screenshot: bytes, 
        target_url: str, 
        navigation_intent: str,
        page_title: str = "",
        visible_text: str = ""
    ) -> Dict[str, Any]:
        """
        Use AI to analyze if the previewed page matches the navigation intent.
        """
        try:
            # Truncate visible text for prompt efficiency
            text_preview = visible_text[:1000] + "..." if len(visible_text) > 1000 else visible_text
            
            system_prompt = f"""
            You are analyzing a webpage to determine if it matches a user's navigation intent.
            
            Navigation Intent: "{navigation_intent}"
            Target URL: {target_url}
            Page Title: {page_title}
            
            Page Text Preview:
            {text_preview}
            
            Look at the webpage screenshot and the provided context to determine:
            1. Does this page match what the user is looking for based on their navigation intent?
            2. What is the main purpose and content of this page?
            3. How confident are you in this assessment?
            
            Consider:
            - Page title and headings visible in the screenshot
            - Main content and overall purpose
            - Navigation elements and page structure
            - Whether this looks like the intended destination
            - How well the page content aligns with the user's stated intent
            
            Be thorough in your analysis and provide clear reasoning.
            """
            
            result = generate_model(
                prompt="Analyze this webpage screenshot and determine if it matches the user's navigation intent.",
                model_object_type=NavigationAnalysis,
                system_prompt=system_prompt,
                image=screenshot,
                reasoning_level="medium"
            )
            
            print(f"[NavigationGoal] AI analysis result: {result}")
            return result.model_dump()
            
        except Exception as e:
            print(f"[NavigationGoal] Error in AI analysis: {e}")
            # Fallback to simple keyword matching
            intent_keywords = navigation_intent.lower().split()
            url_matches = sum(1 for keyword in intent_keywords if keyword in target_url.lower())
            title_matches = sum(1 for keyword in intent_keywords if keyword in page_title.lower())
            text_matches = sum(1 for keyword in intent_keywords if keyword in visible_text.lower())
            
            total_matches = url_matches + title_matches + text_matches
            confidence = min(0.8, total_matches / max(1, len(intent_keywords)))
            
            return {
                "matches_intent": total_matches >= len(intent_keywords) * 0.5,
                "confidence": confidence,
                "reasoning": f"AI analysis failed, used keyword matching. Found {total_matches}/{len(intent_keywords)} matches. Error: {e}",
                "page_summary": f"Page at {target_url} with title '{page_title}'"
            }
    
    def _evaluate_simple_navigation(self, context: GoalContext) -> GoalResult:
        """
        Fallback evaluation using simple URL and text matching.
        """
        current_url = context.current_state.url.lower()
        current_text = (context.current_state.visible_text or "").lower()
        
        # Check URL contains target strings
        url_matches = []
        if self.target_url_contains:
            for target in self.target_url_contains:
                if target.lower() in current_url:
                    url_matches.append(target)
        
        # Check page contains target text
        text_matches = []
        if self.target_page_text:
            for target in self.target_page_text:
                if target.lower() in current_text:
                    text_matches.append(target)
        
        # Determine success
        url_success = not self.target_url_contains or len(url_matches) > 0
        text_success = not self.target_page_text or len(text_matches) > 0
        
        if url_success and text_success:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.7,  # Lower confidence for simple matching
                reasoning=f"Simple navigation check passed - URL matches: {url_matches}, Text matches: {text_matches}",
                evidence={
                    "current_url": current_url,
                    "url_matches": url_matches,
                    "text_matches": text_matches,
                    "evaluation_timing": "post_interaction_simple"
                }
            )
        else:
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=0.8,
                reasoning=f"Simple navigation check incomplete - URL OK: {url_success}, Text OK: {text_success}",
                evidence={
                    "current_url": current_url,
                    "expected_url_contains": self.target_url_contains,
                    "expected_text_contains": self.target_page_text
                }
            )
    
    def get_description(self, context: GoalContext) -> str:
        """
        Generate a detailed description of what this navigation goal is looking for.
        
        The description should include:
        - Goal statement and navigation intent
        - Target criteria (URL patterns, page content)
        - Current page context
        - Preview analysis results (if available)
        - Navigation history and status
        
        Format should be:
        ```
        Navigation goal: [description]
        Navigation intent: [intent]
        Target URL should contain: [criteria]
        Target page should contain: [criteria]
        Current page: [url]
        Preview analysis results: (if available)
          ✅ Preview confirmed target matches intent
          📄 Page summary: [summary]
          🎯 Confidence: [confidence]
          💭 Reasoning: [reasoning]
        Navigation history: [status]
        ```
        """
        description_parts = []
        
        # Main goal description
        description_parts.append(f"Navigation goal: {self.description}")
        description_parts.append(f"Navigation intent: {self.navigation_intent}")
        
        # Add target criteria if specified
        if self.target_url_contains:
            description_parts.append(f"Target URL should contain: {', '.join(self.target_url_contains)}")
        
        if self.target_page_text:
            description_parts.append(f"Target page should contain: {', '.join(self.target_page_text)}")
        
        # Add current page context
        current_url = context.current_state.url
        description_parts.append(f"Current page: {current_url}")
        
        # Add preview results if available
        if self.preview_results:
            description_parts.append("Preview analysis results:")
            if self.preview_results.get("matches_intent"):
                description_parts.append("  ✅ Preview confirmed target matches intent")
                description_parts.append(f"  📄 Page summary: {self.preview_results.get('page_summary', 'N/A')}")
            else:
                description_parts.append("  ❌ Preview shows target does not match intent")
                description_parts.append(f"  📄 Page summary: {self.preview_results.get('page_summary', 'N/A')}")
            description_parts.append(f"  🎯 Confidence: {self.preview_results.get('confidence', 0):.2f}")
            description_parts.append(f"  💭 Reasoning: {self.preview_results.get('reasoning', 'N/A')}")
        
        # Add URL history context
        if len(context.url_history) > 1:
            description_parts.append(f"Navigation history: {len(context.url_history)} pages visited")
            if len(context.url_history) >= 2:
                previous_url = context.url_history[-2]
                if previous_url != current_url:
                    description_parts.append(f"  Previous page: {previous_url}")
                    description_parts.append(f"  Current page: {current_url}")
                    description_parts.append("  ✅ Navigation has occurred")
                else:
                    description_parts.append("  ⏳ No navigation detected yet")
        
        return "\n".join(description_parts)