"""
Target resolver for for loop iterations using vision-based detection.
"""
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Tuple
from playwright.sync_api import Page

from ai_utils import generate_model, generate_text
from .for_models import IterationTargetsResponse, ElementContextResponse, TargetValidationResponse


class TargetResolver:
    """
    Resolves iteration targets using vision-based detection.
    
    This class handles the detection and classification of UI elements
    that can be iterated over in for loops.
    """
    
    def __init__(self):
        self.page: Optional[Page] = None
    
    def set_page(self, page: Page) -> None:
        """Set the page reference for element detection"""
        self.page = page
    
    def resolve_element_targets(self, target_spec: str, screenshot: bytes) -> List[Dict[str, Any]]:
        """
        Resolve element-based targets using vision detection.
        
        Args:
            target_spec: Specification of what to target (e.g., "iOS job", "product", "button")
            screenshot: Full page screenshot for vision analysis
            
        Returns:
            List of resolved targets with rich context
        """
        if not self.page:
            print("⚠️ No page reference available for target resolution")
            return []
        
        print(f"🔍 Detecting {target_spec} targets using vision...")
        
        try:
            # Use vision model to detect and classify targets
            targets_response = self._detect_targets_with_vision(target_spec, screenshot)
            
            if not targets_response or not targets_response.targets:
                print(f"❌ No {target_spec} targets detected")
                return []
            
            print(f"🎯 Vision detected {len(targets_response.targets)} {target_spec} targets")
            print(f"   Confidence: {targets_response.confidence:.2f}")
            print(f"   Reasoning: {targets_response.reasoning}")
            
            # Enrich targets with additional context
            enriched_targets = []
            for i, target in enumerate(targets_response.targets):
                enriched_target = self._enrich_target_context(target, i, screenshot)
                enriched_targets.append(enriched_target)
            
            return enriched_targets
            
        except Exception as e:
            print(f"⚠️ Error in vision-based target detection: {e}")
            # Fallback to basic element detection
            return self._fallback_element_detection(target_spec)
    
    def _detect_targets_with_vision(self, target_spec: str, screenshot: bytes) -> Optional[IterationTargetsResponse]:
        """
        Use vision model to detect iteration targets.
        """
        system_prompt = f"""
        You are analyzing a webpage to identify specific elements for automation.
        
        USER REQUEST: "for each {target_spec}"
        
        Your task:
        1. Identify ALL elements that match the target specification
        2. For each element, extract key information (title, company, location, etc.)
        3. Determine if each element is actionable (clickable, fillable, etc.)
        4. Rank elements by relevance and actionability
        
        Return a structured list of targets with rich context.
        
        Guidelines:
        - Be precise in identifying elements that match the specification
        - Extract meaningful context from each element
        - Focus on elements that are actionable for automation
        - Provide coordinates for precise targeting
        - Consider visual hierarchy and importance
        """
        
        prompt = f"""
        Analyze this webpage and identify all {target_spec} elements.
        
        For each element found, provide:
        - Element type and location (coordinates)
        - Key information (job title, company, product name, etc.)
        - Whether it's actionable (clickable/fillable)
        - Visual description for targeting
        
        Focus on elements that a user would want to iterate over in an automation loop.
        Provide coordinates for precise targeting.
        """
        
        try:
            response = generate_model(
                prompt=prompt,
                model_object_type=IterationTargetsResponse,
                system_prompt=system_prompt,
                image=screenshot,
            )
            
            return response
            
        except Exception as e:
            print(f"⚠️ Vision model error: {e}")
            return None
    
    def _enrich_target_context(self, target: Dict[str, Any], index: int, screenshot: bytes) -> Dict[str, Any]:
        """
        Enrich target with additional context using vision analysis.
        """
        try:
            # Extract coordinates for element analysis
            coordinates = target.get('coordinates', {})
            x = coordinates.get('x', 0)
            y = coordinates.get('y', 0)
            
            if x and y and self.page:
                # Get element info from page
                element_info = self._get_element_info_at_coordinates(x, y)
                
                # Merge vision-detected info with DOM info
                enriched_target = {
                    "index": index,
                    "coordinates": coordinates,
                    "element_info": element_info,
                    "context": target.get('context', {}),
                    "actionable": target.get('actionable', True),
                    "visual_description": target.get('visual_description', ''),
                    "target_id": f"{target.get('element_type', 'element')}_{index}"
                }
                
                return enriched_target
            else:
                # Fallback to basic target info
                return {
                    "index": index,
                    "coordinates": coordinates,
                    "context": target.get('context', {}),
                    "actionable": target.get('actionable', True),
                    "target_id": f"target_{index}"
                }
                
        except Exception as e:
            print(f"⚠️ Error enriching target context: {e}")
            return {
                "index": index,
                "coordinates": target.get('coordinates', {}),
                "context": target.get('context', {}),
                "target_id": f"target_{index}"
            }
    
    def _get_element_info_at_coordinates(self, x: int, y: int) -> Dict[str, Any]:
        """
        Get detailed element information at specific coordinates.
        """
        try:
            element_info = self.page.evaluate(
                """({ x, y }) => {
                    // Convert to client coordinates if needed
                    let cx = x;
                    let cy = y;
                    
                    // Ensure coordinates are within viewport
                    if (cx < 0 || cy < 0 || cx >= window.innerWidth || cy >= window.innerHeight) {
                        return null;
                    }
                    
                    let element = document.elementFromPoint(cx, cy);
                    if (!element) return null;
                    
                    // Get clickable ancestor if available
                    const clickableSelector = 'a,button,[role="link"],[role="button"],input,select,textarea';
                    const clickableAncestor = element.closest ? element.closest(clickableSelector) : null;
                    if (clickableAncestor) {
                        element = clickableAncestor;
                    }
                    
                    const rect = element.getBoundingClientRect();
                    const style = getComputedStyle(element);
                    
                    return {
                        tagName: element.tagName.toLowerCase(),
                        textContent: (element.textContent || '').trim(),
                        innerText: element.innerText ? element.innerText.trim() : '',
                        className: element.className,
                        id: element.id,
                        isClickable: element.tagName.toLowerCase() === 'button' || 
                                    element.tagName.toLowerCase() === 'a' ||
                                    style.cursor === 'pointer',
                        bounds: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        }
                    };
                }""",
                {"x": x, "y": y}
            ) or {}
            
            return element_info
            
        except Exception as e:
            print(f"⚠️ Error getting element info: {e}")
            return {}
    
    def _fallback_element_detection(self, target_spec: str) -> List[Dict[str, Any]]:
        """
        Fallback to basic element detection when vision fails.
        """
        if not self.page:
            return []
        
        try:
            print(f"🔄 Using fallback element detection for {target_spec}")
            
            # Use basic element detection
            elements = self.page.evaluate(
                """() => {
                    // Find common interactive elements
                    const selectors = [
                        'button', 'a', '[role="button"]', '[role="link"]',
                        'input', 'select', 'textarea'
                    ];
                    
                    const elements = [];
                    selectors.forEach(selector => {
                        const found = document.querySelectorAll(selector);
                        found.forEach((el, index) => {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                elements.push({
                                    tagName: el.tagName.toLowerCase(),
                                    textContent: (el.textContent || '').trim(),
                                    className: el.className,
                                    id: el.id,
                                    coordinates: {
                                        x: Math.round(rect.left + rect.width / 2),
                                        y: Math.round(rect.top + rect.height / 2)
                                    },
                                    bounds: {
                                        x: rect.x,
                                        y: rect.y,
                                        width: rect.width,
                                        height: rect.height
                                    }
                                });
                            }
                        });
                    });
                    
                    return elements.slice(0, 10); // Limit to first 10 elements
                }"""
            )
            
            # Convert to target format
            targets = []
            for i, element in enumerate(elements):
                target = {
                    "index": i,
                    "coordinates": element['coordinates'],
                    "context": {
                        "element_type": element['tagName'],
                        "text_content": element['textContent'],
                        "class_name": element['className']
                    },
                    "element_info": element,
                    "actionable": True,
                    "target_id": f"fallback_{i}"
                }
                targets.append(target)
            
            print(f"🔄 Fallback detection found {len(targets)} elements")
            return targets
            
        except Exception as e:
            print(f"⚠️ Fallback element detection failed: {e}")
            return []
    
    def validate_targets(self, targets: List[Dict[str, Any]], screenshot: bytes) -> List[Dict[str, Any]]:
        """
        Validate that targets are still valid and actionable.
        """
        if not targets:
            return []
        
        print(f"🔍 Validating {len(targets)} targets...")
        
        try:
            validation_response = self._validate_targets_with_vision(targets, screenshot)
            
            if validation_response:
                valid_targets = []
                for target, is_valid in zip(targets, validation_response.validations):
                    if is_valid:
                        valid_targets.append(target)
                
                print(f"✅ {len(valid_targets)}/{len(targets)} targets validated")
                return valid_targets
            else:
                print("⚠️ Validation failed, using all targets")
                return targets
                
        except Exception as e:
            print(f"⚠️ Target validation error: {e}")
            return targets
    
    def _validate_targets_with_vision(self, targets: List[Dict[str, Any]], screenshot: bytes) -> Optional[TargetValidationResponse]:
        """
        Use vision to validate target elements.
        """
        system_prompt = """
        You are validating UI elements for automation.
        
        Check each target element to ensure:
        1. It's still visible on the page
        2. It's in the expected location
        3. It's actionable (clickable, fillable, etc.)
        4. It hasn't changed significantly
        5. It matches the original specification
        
        Return validation results for each target.
        """
        
        # Format targets for validation
        target_descriptions = []
        for i, target in enumerate(targets):
            context = target.get('context', {})
            coordinates = target.get('coordinates', {})
            desc = f"Target {i+1}: {context.get('element_type', 'element')} at ({coordinates.get('x', 0)}, {coordinates.get('y', 0)})"
            target_descriptions.append(desc)
        
        prompt = f"""
        Validate these automation targets on the current page:
        
        TARGETS TO VALIDATE:
        {chr(10).join(target_descriptions)}
        
        For each target, determine:
        - Is it still present and visible?
        - Has its appearance changed significantly?
        - Is it still actionable?
        - Does it still match the original criteria?
        
        Return validation results for each target.
        """
        
        try:
            response = generate_model(
                prompt=prompt,
                model_object_type=TargetValidationResponse,
                system_prompt=system_prompt,
                image=screenshot,
            )
            
            return response
            
        except Exception as e:
            print(f"⚠️ Validation model error: {e}")
            return None
