"""
Contextual command engine for rewriting automation commands for specific targets.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from ai_utils import generate_text, generate_model


class CommandRewriteResponse(BaseModel):
    """Response model for command rewriting"""
    rewritten_command: str = Field(description="The rewritten command")
    reasoning: str = Field(description="Explanation of the rewrite")
    confidence: float = Field(description="Confidence in the rewrite (0-1)")


class ContextualCommandEngine:
    """
    Rewrites automation commands to be specific to target elements.
    
    This class takes generic commands like "click it" or "apply to it"
    and rewrites them to be specific to the target element being processed.
    """
    
    def __init__(self):
        self.rewrite_cache: Dict[str, str] = {}
    
    def rewrite_command(self, original_command: str, target: Dict[str, Any], screenshot: Optional[bytes] = None) -> str:
        """
        Rewrite a command to be specific to the target element.
        
        Args:
            original_command: The original generic command
            target: The target element with context
            screenshot: Optional screenshot for visual context
            
        Returns:
            Rewritten command specific to the target
        """
        # Check cache first
        cache_key = f"{original_command}_{target.get('target_id', 'unknown')}"
        if cache_key in self.rewrite_cache:
            return self.rewrite_cache[cache_key]
        
        print(f"🔄 Rewriting command: '{original_command}' for target {target.get('target_id', 'unknown')}")
        
        try:
            # Try AI-powered rewriting first
            rewritten = self._rewrite_with_ai(original_command, target, screenshot)
            
            # Cache the result
            self.rewrite_cache[cache_key] = rewritten
            
            print(f"   → Rewritten to: '{rewritten}'")
            return rewritten
            
        except Exception as e:
            print(f"⚠️ AI rewriting failed: {e}, using fallback")
            # Fallback to rule-based rewriting
            return self._rewrite_with_rules(original_command, target)
    
    def _rewrite_with_ai(self, original_command: str, target: Dict[str, Any], screenshot: Optional[bytes] = None) -> str:
        """
        Use AI to intelligently rewrite commands for specific targets.
        """
        system_prompt = """
        You are rewriting automation commands to be specific to a target element.
        
        Your task:
        1. Take a generic command ("click it", "fill it", "apply to it")
        2. Rewrite it to be specific to the target element
        3. Use the element's visual context and information
        4. Make the command unambiguous and actionable
        5. Preserve the original intent while making it specific
        
        Guidelines:
        - Use specific details from the target context
        - Make commands clear and unambiguous
        - Consider the element's visual appearance and text
        - Maintain natural language that the automation system can understand
        - Focus on what the user wants to accomplish with this specific element
        """
        
        # Extract target context
        context = target.get('context', {})
        coordinates = target.get('coordinates', {})
        element_info = target.get('element_info', {})
        
        # Build context description
        context_parts = []
        if 'job_title' in context:
            context_parts.append(f"Job: {context['job_title']}")
        if 'company' in context:
            context_parts.append(f"Company: {context['company']}")
        if 'location' in context:
            context_parts.append(f"Location: {context['location']}")
        if 'product_name' in context:
            context_parts.append(f"Product: {context['product_name']}")
        if 'price' in context:
            context_parts.append(f"Price: {context['price']}")
        
        element_text = element_info.get('textContent', '') or element_info.get('innerText', '')
        if element_text:
            context_parts.append(f"Text: {element_text}")
        
        context_description = ", ".join(context_parts) if context_parts else "Generic element"
        
        prompt = f"""
        Rewrite this automation command for the specific target:
        
        ORIGINAL COMMAND: "{original_command}"
        
        TARGET ELEMENT:
        - Context: {context_description}
        - Coordinates: ({coordinates.get('x', 0)}, {coordinates.get('y', 0)})
        - Element Type: {element_info.get('tagName', 'unknown')}
        - Target ID: {target.get('target_id', 'unknown')}
        
        Rewrite the command to be specific to this target element.
        Use the element's information to make it clear and actionable.
        
        Examples of good rewrites:
        - "click it" → "click: the 'iOS Developer' job at Apple Inc"
        - "fill it" → "type: user@example.com in the email field"
        - "apply to it" → "click: the apply button for the Senior iOS Engineer position at Google"
        - "add to cart" → "click: the add to cart button for iPhone 15 Pro"
        
        Return only the rewritten command, no explanations.
        """
        
        try:
            if screenshot:
                # Use vision model if screenshot is available
                response = generate_model(
                    prompt=prompt,
                    model_object_type=CommandRewriteResponse,
                    system_prompt=system_prompt,
                    image=screenshot,
                    reasoning_level="medium"
                )
                return response.rewritten_command
            else:
                # Use text-only model
                response = generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    reasoning_level="medium"
                )
                return response.strip()
                
        except Exception as e:
            print(f"⚠️ AI command rewriting error: {e}")
            raise e
    
    def _rewrite_with_rules(self, original_command: str, target: Dict[str, Any]) -> str:
        """
        Fallback rule-based command rewriting.
        """
        context = target.get('context', {})
        element_info = target.get('element_info', {})
        
        # Extract key information
        job_title = context.get('job_title', '')
        company = context.get('company', '')
        product_name = context.get('product_name', '')
        element_text = element_info.get('textContent', '') or element_info.get('innerText', '')
        
        # Rule-based rewriting
        original_lower = original_command.lower()
        
        if "click it" in original_lower or "click on it" in original_lower:
            if job_title and company:
                return f"click on the '{job_title}' job at {company}"
            elif product_name:
                return f"click on the {product_name}"
            elif element_text:
                return f"click on the {element_text}"
            else:
                return f"click on the element"
        
        elif "apply to it" in original_lower:
            if job_title and company:
                return f"click apply for the {job_title} position at {company}"
            elif job_title:
                return f"click apply for the {job_title} job"
            else:
                return "click apply"
        
        elif "add to cart" in original_lower:
            if product_name:
                return f"click add to cart for {product_name}"
            else:
                return "click add to cart"
        
        elif "fill it" in original_lower:
            field_type = context.get('field_type', 'field')
            return f"fill the {field_type}"
        
        elif "select it" in original_lower:
            option_text = element_text or context.get('option_text', 'option')
            return f"select the {option_text} option"
        
        else:
            # Generic fallback
            if element_text:
                return f"{original_command} for {element_text}"
            else:
                return original_command
    
    def clear_cache(self) -> None:
        """Clear the command rewrite cache"""
        self.rewrite_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_commands": len(self.rewrite_cache),
            "cache_hits": 0,  # Would need to track this
            "cache_misses": 0  # Would need to track this
        }
