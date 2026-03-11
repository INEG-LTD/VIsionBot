from __future__ import annotations

from dataclasses import dataclass, field
import html
from pathlib import Path
from typing import Any, Optional
import re


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    path: Path
    license: Optional[str] = None
    compatibility: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def discover_skills(directories: list[str]) -> list[SkillMeta]:
    """Discover skills from directory roots containing folders with SKILL.md files."""
    discovered: list[SkillMeta] = []
    seen_paths: set[str] = set()

    for raw_directory in directories or []:
        directory_text = str(raw_directory or "").strip()
        if not directory_text:
            continue
        root = Path(directory_text).expanduser()
        if not root.is_dir():
            continue

        candidates: list[Path] = [root]
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    candidates.append(child)
        except Exception:
            pass

        for skill_dir in candidates:
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                resolved_dir = skill_dir.resolve()
            except Exception:
                resolved_dir = skill_dir
            path_key = str(resolved_dir)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            try:
                raw = skill_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            frontmatter, body = _split_frontmatter(raw)

            name = str(frontmatter.get("name") or resolved_dir.name).strip() or resolved_dir.name
            description = _extract_description(frontmatter, body)
            license_name = _normalize_optional_text(frontmatter.get("license"))
            compatibility = _normalize_compatibility(frontmatter.get("compatibility"))
            metadata = {
                str(k): v
                for k, v in frontmatter.items()
                if str(k) not in {"name", "description", "license", "compatibility"}
            }

            discovered.append(
                SkillMeta(
                    name=name,
                    description=description,
                    path=resolved_dir,
                    license=license_name,
                    compatibility=compatibility,
                    metadata=metadata,
                )
            )

    discovered.sort(key=lambda item: (item.name.casefold(), str(item.path)))
    return discovered


def load_skill_body(
    skill: SkillMeta,
    max_chars: Optional[int] = None,
    *,
    return_stats: bool = False,
) -> Any:
    """Load full markdown body (after frontmatter) for an active skill."""
    skill_file = skill.path / "SKILL.md"
    raw = skill_file.read_text(encoding="utf-8", errors="replace")
    _, body = _split_frontmatter(raw)
    text = body.strip()
    if not text:
        text = "(Skill has no body instructions.)"
    original_chars = len(text)
    retained_chars = original_chars
    was_truncated = False

    cap = int(max_chars or 0)
    if cap > 0 and len(text) > cap:
        was_truncated = True
        retained_text = text[:cap].rstrip()
        retained_chars = len(retained_text)
        text = f"{retained_text}\n\n[Skill body truncated to {cap} characters.]"

    if return_stats:
        return text, {
            "original_chars": int(original_chars),
            "retained_chars": int(retained_chars),
            "truncated": bool(was_truncated),
            "cap_chars": int(cap),
        }
    return text


def format_skills_catalog(skills: list[SkillMeta]) -> str:
    """Render lightweight metadata for all discovered skills."""
    lines: list[str] = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{html.escape(skill.name)}</name>")
        lines.append(f"    <description>{html.escape(skill.description)}</description>")
        if skill.license:
            lines.append(f"    <license>{html.escape(skill.license)}</license>")
        if skill.compatibility:
            compat = ", ".join(skill.compatibility)
            lines.append(f"    <compatibility>{html.escape(compat)}</compatibility>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_active_skill(skill: SkillMeta, body: str) -> str:
    """Render active skill with full body instructions and directory context."""
    safe_body = str(body or "").rstrip()
    if not safe_body:
        safe_body = "(Skill has no body instructions.)"
    directory_header = f"Skill directory: {skill.path}"
    payload = f"{directory_header}\n\n{safe_body}".replace("]]>", "]]]]><![CDATA[>")

    return (
        "<active_skill>\n"
        f"  <name>{html.escape(skill.name)}</name>\n"
        f"  <description>{html.escape(skill.description)}</description>\n"
        f"  <skill_directory>{html.escape(str(skill.path))}</skill_directory>\n"
        "  <instructions><![CDATA[\n"
        f"{payload}\n"
        "]]></instructions>\n"
        "</active_skill>"
    )


def _extract_description(frontmatter: dict[str, Any], body: str) -> str:
    fm_description = _normalize_optional_text(frontmatter.get("description"))
    if fm_description:
        return fm_description
    for raw_line in str(body or "").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        if text.startswith("#"):
            text = text.lstrip("#").strip()
        if text:
            return text
    return "No description provided."


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    lines = str(content or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, str(content or "")

    closing_idx: Optional[int] = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break
    if closing_idx is None:
        return {}, str(content or "")

    frontmatter_text = "\n".join(lines[1:closing_idx])
    body = "\n".join(lines[closing_idx + 1 :])
    metadata = _parse_frontmatter(frontmatter_text)
    return metadata, body


def _parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(frontmatter_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return _parse_frontmatter_fallback(frontmatter_text)


def _parse_frontmatter_fallback(frontmatter_text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_list_key: Optional[str] = None

    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- ") and current_list_key and isinstance(metadata.get(current_list_key), list):
            metadata[current_list_key].append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in line:
            current_list_key = None
            continue

        key_text, value_text = line.split(":", 1)
        key = key_text.strip()
        if not key:
            current_list_key = None
            continue
        value = value_text.strip()
        if not value:
            metadata[key] = []
            current_list_key = key
            continue
        metadata[key] = _parse_scalar(value)
        current_list_key = None

    return metadata


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in re.split(r"\s*,\s*", inner)]

    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except Exception:
            return text
    return text


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_compatibility(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    raw_items = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)
