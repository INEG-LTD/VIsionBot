"""
Form processing utilities for the vision bot.
"""

from typing import Dict, Any, List


class FormUtils:
    """Utility class for form processing operations."""
    
    @staticmethod
    def form_status_js() -> str:
        """JavaScript code to check form status across the page."""
        return """
        () => {
        const visible = (el) => {
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
        };
        
        const blockers = [];
        const groups = {};
        const iframes = [];
        
        // Collect iframe info first
        for (const iframe of document.querySelectorAll('iframe')) {
            iframes.push({
                src: iframe.src,
                visible: visible(iframe),
                sameOrigin: try { iframe.contentWindow.location.href; true; } catch { false; }
            });
        }
        
        // Walk all form controls
        for (const el of document.querySelectorAll('input, select, textarea, [contenteditable="true"]')) {
            if (!visible(el)) continue;
            
            const tag = el.tagName;
            const required = el.hasAttribute('required') || 
                           el.getAttribute('aria-required') === 'true' ||
                           el.closest('[data-required="true"]') ||
                           /\\*\\s*$/.test(el.previousElementSibling?.textContent || '');
            
            if (!required) continue;
            
            const addBlocker = (elem, reason) => {
                blockers.push({
                    tag: elem.tagName,
                    type: elem.type || 'text',
                    name: elem.name || elem.id || 'unnamed',
                    reason: reason,
                    text: elem.previousElementSibling?.textContent?.trim() || 
                          elem.placeholder || 
                          elem.getAttribute('aria-label') || 
                          'required field'
                });
            };
            
            // Handle different input types
            if (tag === 'INPUT') {
                const type = el.type || 'text';
                if (type === 'checkbox' || type === 'radio') {
                    const name = el.name;
                    if (!groups[name]) groups[name] = [];
                    groups[name].push(el);
                    continue;
                }
                
                const val = (el.value || '').trim();
                if (!val) addBlocker(el, 'required empty');
                continue;
            }
            
            if (tag === 'TEXTAREA') {
                const val = (el.value || '').trim();
                if (!val) addBlocker(el, 'required empty');
                continue;
            }
            
            if (tag === 'SELECT') {
                const val = el.value;
                const opt = el.selectedOptions && el.selectedOptions[0];
                const looksPlaceholder = !!(opt && (
                    opt.disabled ||
                    /select|choose|please|--/i.test(opt.textContent || '')
                ));
                if (!val || val === '' || looksPlaceholder) addBlocker(el, 'required unselected');
                continue;
            }
            
            // ARIA combobox / custom selects
            if (el.getAttribute && el.getAttribute('role') === 'combobox') {
                const val = (el.getAttribute('aria-valuetext') ||
                            el.getAttribute('aria-activedescendant') ||
                            el.textContent || '').trim();
                if (!val || /select|choose|please/i.test(val)) addBlocker(el, 'required unselected (combobox)');
                continue;
            }
            
            // Contenteditable rich inputs
            if (el.getAttribute && el.getAttribute('contenteditable') === 'true') {
                const val = (el.textContent || '').trim();
                if (!val) addBlocker(el, 'required empty (contenteditable)');
                continue;
            }
        }
        
        // Resolve checkbox/radio groups
        for (const name in groups) {
            const group = groups[name].filter(visible);
            if (group.length && !group.some(g => g.checked)) {
                addBlocker(group[0], 'required group unchecked');
            }
        }
        
        const formsVisible = Array.from(document.querySelectorAll('form')).some(visible);
        const hasErrors = !!document.querySelector('[aria-invalid="true"], .error, .has-error, [role="alert"]');
        const submitButtons = Array.from(document.querySelectorAll('button,[type="submit"],[role="button"]'))
            .filter(visible);
        const hasDisabledSubmit = submitButtons.some(
            b => /submit|apply|send/i.test(b.textContent || b.value || '') &&
                (b.disabled || b.getAttribute('aria-disabled') === 'true')
        );
        
        return {
            requiredEmpty: blockers.length,
            blockers: blockers.slice(0, 6), // trim to keep payload small
            formsVisible,
            hasErrors,
            hasDisabledSubmit,
            iframes, // only meaningful in top document
        };
        }
        """
    
    @staticmethod
    def collect_form_status_across_contexts(page) -> Dict[str, Any]:
        """Inspect top page and all *same-origin* iframes. Flag visible cross-origin ATS iframes."""
        # Top document
        top = page.evaluate(FormUtils.form_status_js())
        
        required = int(top.get("requiredEmpty") or 0)
        blockers = list(top.get("blockers") or [])
        forms_visible = bool(top.get("formsVisible"))
        has_errors = bool(top.get("hasErrors"))
        has_disabled_submit = bool(top.get("hasDisabledSubmit"))
        
        # Detect visible cross-origin ATS iframes (we can't inspect inside them)
        cross_origin_ats = []
        for f in (top.get("iframes") or []):
            if not f.get("visible"):
                continue
            src = (f.get("src") or "").lower()
            same = bool(f.get("sameOrigin"))
            if (not same) and any(domain in src for domain in ["workday", "lever", "greenhouse", "bamboohr", "icims"]):
                cross_origin_ats.append(src)
        
        # Walk same-origin frames and aggregate
        for frame in page.frames:
            # Skip the main frame; Playwright's main frame URL equals page.url
            if frame == page.main_frame:
                continue
            try:
                data = frame.evaluate(FormUtils.form_status_js())
            except Exception:
                # Cross-origin (can't eval)
                continue
            required += int(data.get("requiredEmpty") or 0)
            blockers += list(data.get("blockers") or [])
            forms_visible = forms_visible or bool(data.get("formsVisible"))
            has_errors = has_errors or bool(data.get("hasErrors"))
            has_disabled_submit = has_disabled_submit or bool(data.get("hasDisabledSubmit"))
        
        return {
            "requiredEmpty": required,
            "blockers": blockers,
            "formsVisible": forms_visible,
            "hasErrors": has_errors,
            "hasDisabledSubmit": has_disabled_submit,
            "crossOriginATS": cross_origin_ats,
        }
    
    @staticmethod
    def is_required_control_at_point(page, x: int, y: int) -> bool:
        """Check if there's a required form control at the specified coordinates."""
        try:
            element = page.element_from_point(x, y)
            if not element:
                return False
            
            # Check if it's a form control
            tag = element.tag_name.lower()
            if tag not in ['input', 'select', 'textarea']:
                return False
            
            # Check if it's required
            required = element.get_attribute('required') or element.get_attribute('aria-required') == 'true'
            if not required:
                # Check if it has a required indicator (like *)
                parent = element.query_selector('xpath=..')
                if parent:
                    text = parent.text_content or ''
                    required = text.strip().endswith('*')
            
            return required
        except Exception:
            return False
