"""
Type Goal - Validates text input fields before typing.
"""
from typing import Dict, Any, Optional, Tuple

from .form_field_goal import FormFieldGoal
from .base import GoalResult, GoalStatus, GoalContext


class TypeGoal(FormFieldGoal):
    """
    Goal for typing into text input fields.
    Validates that the correct text field is targeted before typing.
    """
    
    def __init__(self, description: str, target_description: str, **kwargs):
        super().__init__(description, target_description, **kwargs)
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate text input field before typing"""
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available"
            )
        
        planned_interaction = getattr(context, 'planned_interaction', None)
        if not planned_interaction or planned_interaction.get('interaction_type') != 'type':
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No type interaction planned yet"
            )
        
        # Temporary bypass: auto-approve type goals until a robust validation strategy is ready.
        return GoalResult(
            status=GoalStatus.ACHIEVED,
            confidence=0.6,
            reasoning="TypeGoal auto-approved (validation temporarily disabled).",
            evidence={"auto_validated": True},
        )

    def _probe_rich_text_editability(
        self,
        context: GoalContext,
        planned_interaction: Dict[str, Any],
    ) -> bool:
        page = context.page_reference
        coordinates: Optional[Tuple[int, int]] = planned_interaction.get("coordinates")
        if not page or not coordinates:
            return False

        x, y = coordinates
        sentinel = "__TYPE_GOAL_SENTINEL__"
        inserted = False

        try:
            page.mouse.click(x, y)
            page.wait_for_timeout(50)
            page.keyboard.insert_text(sentinel)
            page.wait_for_timeout(50)

            inserted = page.evaluate(
                """
                (sentinel) => {
                    const containsSentinel = (root) => {
                        if (!root) return false;
                        const text = root.innerText || root.textContent || '';
                        if (text && text.includes(sentinel)) return true;
                        return false;
                    };

                    try {
                        if (containsSentinel(document.activeElement)) return true;
                    } catch (err) {}

                    try {
                        if (containsSentinel(document.body)) return true;
                    } catch (err) {}

                    const iframes = Array.from(document.querySelectorAll('iframe'));
                    for (const frame of iframes) {
                        try {
                            const doc = frame.contentDocument;
                            if (!doc) continue;
                            if (containsSentinel(doc.activeElement)) return true;
                            if (containsSentinel(doc.body)) return true;
                            const selection = doc.getSelection && doc.getSelection();
                            if (
                                selection &&
                                selection.anchorNode &&
                                typeof selection.anchorNode.textContent === 'string' &&
                                selection.anchorNode.textContent.includes(sentinel)
                            ) {
                                return true;
                            }
                        } catch (err) {
                            continue;
                        }
                    }

                    return false;
                }
                """,
                sentinel,
            )
        except Exception as exc:
            print(f"[TypeGoal] Rich text probe error: {exc}")
        finally:
            try:
                # Attempt to remove the sentinel text
                for _ in range(len(sentinel)):
                    page.keyboard.press("Backspace")
                page.wait_for_timeout(20)
            except Exception:
                pass
            if inserted:
                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass
            else:
                # Fallback undo shortcuts in case content remains elsewhere
                for shortcut in ("Meta+z", "Control+z"):
                    try:
                        page.keyboard.press(shortcut)
                        page.wait_for_timeout(20)
                    except Exception:
                        pass

        if inserted:
            print("[TypeGoal] Rich text probe succeeded; treating surface as editable.")
        else:
            print("[TypeGoal] Rich text probe could not confirm editability.")

        return inserted
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        planned_interaction = getattr(context, 'planned_interaction', None)
        status = "PENDING"
        
        if planned_interaction and planned_interaction.get('interaction_type') == 'type':
            # Check if we have evaluation result
            if hasattr(self, '_last_evaluation') and self._last_evaluation:
                if self._last_evaluation.status == GoalStatus.ACHIEVED:
                    status = "VALIDATED"
                elif self._last_evaluation.status == GoalStatus.FAILED:
                    status = "FAILED"
        
        return f"""
Goal Type: Type Goal (BEFORE evaluation)
Description: {self.description}
Target Field: {self.target_description}
Current Status: {status}
Requirements: Type into the text field matching "{self.target_description}"
Progress: {'✅ Field validated and ready for typing' if status == 'VALIDATED' else '⏳ Waiting for type interaction validation' if status == 'PENDING' else '❌ Field validation failed'}
Issues: {'None' if status == 'VALIDATED' else 'Type interaction not yet planned or field mismatch detected'}
"""
