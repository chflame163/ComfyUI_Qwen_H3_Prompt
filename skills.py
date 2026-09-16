from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

SKILL_ROOT = Path(__file__).with_name("skills")
CUSTOM_SKILL_ROOT = Path(__file__).with_name("custom_skills")
LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt.skills")
SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class SkillSpec:
    """A discovered prompt-only Skill available to the local node."""

    id: str
    path: Path
    description: str
    source: str
    display_name: str | None = None
    version: str | None = None
    tags: tuple[str, ...] = ()


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _list_values(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(item for item in value if item.strip())
    if not isinstance(value, str) or not value.strip():
        return ()
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        return tuple(
            item
            for item in (_scalar(part) for part in value.split(","))
            if item.strip()
        )
    return (value,)


def _front_matter(path: Path) -> dict[str, str]:
    """Read the small subset of YAML front matter needed for Skill discovery."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}

    end = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].strip() == "---"
            and not lines[index].startswith((" ", "\t"))
        ),
        None,
    )
    if end is None:
        return {}

    values: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        match = re.match(
            r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$",
            line,
        )
        if not match or match.group("indent"):
            index += 1
            continue

        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value.startswith(("|", ">")):
            block: list[str] = []
            cursor = index + 1
            while cursor < end:
                candidate = lines[cursor]
                if candidate.strip() and not candidate.startswith((" ", "\t")):
                    break
                block.append(candidate)
                cursor += 1
            non_empty = [line for line in block if line.strip()]
            indent = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
            block = [line[indent:] if line.strip() else "" for line in block]
            values[key] = ("\n" if value.startswith("|") else " ").join(block).strip()
            index = cursor
            continue
        if value:
            values[key] = _scalar(value)
        index += 1
    return values


def _meta_yaml(path: Path) -> dict[str, str | tuple[str, ...]]:
    """Read simple scalar and list metadata from the bundled meta.yaml convention."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: dict[str, str | tuple[str, ...]] = {}
    index = 0
    while index < len(lines):
        match = re.match(r"^(?P<key>[A-Za-z0-9_-]+):\s*(?P<value>.*)$", lines[index])
        if not match:
            index += 1
            continue
        key = match.group("key").replace("-", "_").casefold()
        value = match.group("value").strip()
        if value:
            values[key] = _scalar(value)
            index += 1
            continue

        items: list[str] = []
        cursor = index + 1
        while cursor < len(lines):
            item = re.match(r"^\s*-\s*(.+?)\s*$", lines[cursor])
            if not item:
                break
            items.append(_scalar(item.group(1)))
            cursor += 1
        if items:
            values[key] = tuple(items)
        index = cursor
    return values


def _metadata(skill_root: Path) -> tuple[str, str, str | None, str | None, tuple[str, ...]]:
    front = _front_matter(skill_root / "SKILL.md")
    meta = _meta_yaml(skill_root / "meta.yaml")
    skill_id = front.get("name", "").strip() or skill_root.name
    description = (
        front.get("description", "").strip()
        or str(meta.get("summary_en", "")).strip()
        or str(meta.get("desc_en", "")).strip()
        or skill_id
    )
    description = " ".join(description.split())
    display_name = (
        front.get("display_name", "").strip()
        or front.get("display_name_en", "").strip()
        or front.get("display_name_zh", "").strip()
        or str(meta.get("display_name_en", "")).strip()
        or str(meta.get("display_name_zh", "")).strip()
        or None
    )
    version = (
        front.get("version", "").strip()
        or str(meta.get("version", "")).strip()
        or None
    )
    raw_tags: list[str] = []
    for values in (
        _list_values(front.get("tags")),
        _list_values(front.get("tag_en")),
        _list_values(front.get("tag_cn")),
        _list_values(meta.get("tags")),
        _list_values(meta.get("tag_en")),
        _list_values(meta.get("tag_cn")),
        _list_values(meta.get("complete_tags_en")),
        _list_values(meta.get("complete_tags_cn")),
    ):
        raw_tags.extend(values)
    return skill_id, description, display_name, version, tuple(dict.fromkeys(raw_tags))


def _discover_skills(root: Path, source: str) -> list[SkillSpec]:
    if not root.is_dir():
        return []
    resolved_root = root.resolve()
    discovered: list[SkillSpec] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not candidate.is_dir():
            continue
        skill_path = candidate.resolve()
        if skill_path.parent != resolved_root:
            LOGGER.warning("Ignoring Skill outside %s: %s", resolved_root, candidate)
            continue
        skill_file = skill_path / "SKILL.md"
        if not skill_file.is_file() or skill_file.resolve().parent != skill_path:
            LOGGER.warning("Ignoring Skill without SKILL.md: %s", skill_path)
            continue
        skill_id, description, display_name, version, tags = _metadata(skill_path)
        if not SKILL_ID_RE.fullmatch(skill_id) or skill_id == "auto":
            LOGGER.warning("Ignoring Skill with invalid id %r: %s", skill_id, skill_path)
            continue
        discovered.append(
            SkillSpec(
                id=skill_id,
                path=skill_path,
                description=description,
                source=source,
                display_name=display_name,
                version=version,
                tags=tags,
            )
        )
    return discovered


def discover_skill_registry(
    builtin_root: Path | None = None,
    custom_root: Path | None = None,
) -> tuple[SkillSpec, ...]:
    """Discover bundled Skills first, then non-overriding user Skills."""
    builtin_root = SKILL_ROOT if builtin_root is None else builtin_root
    custom_root = CUSTOM_SKILL_ROOT if custom_root is None else custom_root
    registry: list[SkillSpec] = []
    seen: set[str] = set()
    candidates = [
        *_discover_skills(builtin_root, "builtin"),
        *_discover_skills(custom_root, "custom"),
    ]
    for spec in candidates:
        key = spec.id.casefold()
        if key in seen:
            LOGGER.warning("Ignoring duplicate Skill id %r from %s", spec.id, spec.path)
            continue
        seen.add(key)
        registry.append(spec)
    return tuple(registry)


SKILL_REGISTRY = discover_skill_registry()
SKILL_NAMES = tuple(spec.id for spec in SKILL_REGISTRY)
SKILL_DESCRIPTIONS = {spec.id: spec.description for spec in SKILL_REGISTRY}
SKILL_BY_ID = {spec.id.casefold(): spec for spec in SKILL_REGISTRY}
DEFAULT_SKILL_ID = (
    "h3-prompt-writing"
    if "h3-prompt-writing" in SKILL_NAMES
    else (SKILL_NAMES[0] if SKILL_NAMES else "")
)

BASE_FIELDS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
REF_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)
TIMESTAMP_RE = re.compile(r"\bAt\s+(\d{2}):(\d{2})\.(\d{3})\b")
INTERACTIVE_OUTPUT_PATTERNS = (
    re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:required\s+)?"
        r"(?:clarifications?|questions?|next steps?|start gate|approval|confirmation)\b"
    ),
    re.compile(r"(?i)\bplease\s+(?:confirm|reply|provide|upload|choose|answer|let\s+me\s+know)\b"),
    re.compile(r"(?i)\b(?:do|would|could|can)\s+you\b"),
    re.compile(r"(?i)\bshould\s+(?:the|we|i)\b[^.\n]*\?"),
    re.compile(r"(?i)\b(?:confirm|reply)\s+[\"']?(?:confirm|go\s+ahead|continue)\b"),
    re.compile(r"(?i)\bto\s+proceed\s+with\s+(?:generation|the\s+next\s+step)\b"),
    re.compile(r"(?i)\bawait(?:ing)?\s+(?:your\s+)?(?:confirmation|approval|reply)\b"),
    re.compile(r"请(?:确认|回复|提供|上传|选择|回答|告知)"),
    re.compile(r"(?:你|您)(?:是否|希望|想要)"),
    re.compile(r"等待(?:你|您|用户)?(?:确认|回复)"),
    re.compile(r"确认后(?:再|即可|将)"),
)
SILENCE_REQUEST_PATTERNS = (
    re.compile(
        r"(?i)\b(?:completely silent|complete silence|silent video|muted video|"
        r"no audio|no sound at all|without any (?:audio|sound))\b"
    ),
    re.compile(r"(?:完全静音|全程静音|静音视频|视频静音|无声视频|不要任何声音|不需要任何声音)"),
)


def detect_h3_mode(image_count: int, video_count: int) -> str | None:
    """Resolve modes that do not require understanding the user's image intent."""
    if not 0 <= image_count <= 9:
        raise ValueError("Reference image count must be between 0 and 9")
    if not 0 <= video_count <= 3:
        raise ValueError("Reference video count must be between 0 and 3")
    if video_count or image_count > 2:
        return "ref2va"
    if image_count == 0:
        return "t2va"
    return None


def mode_router_prompt(user_prompt: str, image_count: int) -> list[dict[str, str]]:
    if image_count not in {1, 2}:
        raise ValueError("Automatic image mode routing requires one or two images")
    if image_count == 1:
        choices = "i2va, l2va, ref2va"
        rules = (
            "Choose i2va only when the request clearly uses Picture 1 as the opening or first frame. "
            "Choose l2va only when it clearly uses Picture 1 as the ending or last frame. "
            "Otherwise choose ref2va, including character, product, scene, style, composition, or other general references."
        )
    else:
        choices = "fl2va, ref2va"
        rules = (
            "Choose fl2va only when the request clearly assigns Picture 1 as the opening/first frame and "
            "Picture 2 as the ending/last frame, or explicitly asks to transition/interpolate from the first "
            "image to the second. Otherwise choose ref2va, including when both pictures are general references."
        )
    system = (
        "Classify the MiniMax H3 image input mode from the user's stated intent. "
        f"There are exactly {image_count} connected pictures. Return exactly one lowercase id from: {choices}. "
        "Return no punctuation or explanation. Do not assume that a picture is a keyframe merely because only "
        f"one or two pictures are connected. {rules}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def parse_mode_selection(text: str, image_count: int) -> str:
    allowed = {1: ("i2va", "l2va", "ref2va"), 2: ("fl2va", "ref2va")}
    choices = allowed.get(image_count)
    if choices is None:
        raise ValueError("Automatic image mode routing requires one or two images")
    normalized = text.strip().lower()
    if normalized in choices:
        return normalized
    matches = [mode for mode in choices if re.search(rf"\b{mode}\b", normalized)]
    return matches[0] if len(matches) == 1 else "ref2va"


def explicitly_requests_silence(user_prompt: str) -> bool:
    return any(pattern.search(user_prompt) for pattern in SILENCE_REQUEST_PATTERNS)


def router_prompt(
    user_prompt: str,
    mode: str,
    asset_summary: str,
    skills: tuple[SkillSpec, ...] | None = None,
) -> list[dict[str, str]]:
    available = skills if skills is not None else SKILL_REGISTRY
    catalog = "\n".join(f"- {spec.id}: {spec.description}" for spec in available)
    system = (
        "Select exactly one available H3 skill for the request. Return only its exact skill id, "
        "with no punctuation or explanation. Prefer h3-prompt-writing for direct prompt rewriting "
        "when no specialized production workflow clearly applies.\n\n" + catalog
    )
    user = f"Detected H3 mode: {mode}\nConnected assets: {asset_summary}\nRequest:\n{user_prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_skill_selection(
    text: str,
    skills: tuple[SkillSpec, ...] | None = None,
) -> str:
    available = skills if skills is not None else SKILL_REGISTRY
    by_id = {spec.id.casefold(): spec.id for spec in available}
    normalized = text.strip().lower()
    if normalized in by_id:
        return by_id[normalized]
    matches = [
        spec.id
        for spec in available
        if re.search(
            rf"(?<![a-z0-9_.-]){re.escape(spec.id.casefold())}(?![a-z0-9_.-])",
            normalized,
        )
    ]
    if not matches:
        matches = [
            spec.id
            for spec in available
            if re.search(
                rf"(?<![a-z0-9_-]){re.escape(spec.id.casefold())}(?![a-z0-9_-])",
                normalized,
            )
        ]
    if len(matches) == 1:
        return matches[0]
    fallback = next((spec.id for spec in available if spec.id == DEFAULT_SKILL_ID), None)
    return fallback or (available[0].id if available else "")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _skill_spec(skill: str) -> SkillSpec:
    spec = SKILL_BY_ID.get(skill.casefold())
    if spec is None:
        raise ValueError(f"Unknown skill: {skill}")
    return spec


def _skill_path(skill: str) -> Path:
    spec = _skill_spec(skill)
    root = spec.path.resolve()
    allowed_roots = (SKILL_ROOT.resolve(), CUSTOM_SKILL_ROOT.resolve())
    if not any(root.parent == allowed_root for allowed_root in allowed_roots):
        raise FileNotFoundError(f"Skill path is outside the configured Skill roots: {skill}")
    skill_file = root / "SKILL.md"
    if (
        not root.is_dir()
        or not skill_file.is_file()
        or skill_file.resolve().parent != root
    ):
        raise FileNotFoundError(f"Skill is missing SKILL.md: {skill}")
    return root


def skill_instructions(skill: str, mode: str, max_chars: int = 72000) -> str:
    skill = _skill_spec(skill).id
    root = _skill_path(skill)

    parts = [f"# Selected H3 Skill: {skill}\n\n{_read(root / 'SKILL.md')}"]
    if skill == "h3-prompt-writing":
        guide = "ref-en.txt" if mode == "ref2va" else "base-en.txt"
        parts.append(f"# Required H3 reference: {guide}\n\n{_read(root / 'references' / guide)}")
    else:
        used = len(parts[0])
        for path in (
            sorted((root / "references").glob("**/*")) if (root / "references").is_dir() else ()
        ):
            resolved_path = path.resolve()
            if (
                not path.is_file()
                or path.suffix.lower() not in {".md", ".txt"}
                or (
                    resolved_path.parent != root
                    and root not in resolved_path.parents
                )
            ):
                continue
            text = _read(resolved_path)
            addition = f"# Skill reference: {path.relative_to(root).as_posix()}\n\n{text}"
            if used + len(addition) > max_chars:
                break
            parts.append(addition)
            used += len(addition)
    return "\n\n".join(parts)


def system_prompt(skill: str, mode: str, duration: float) -> str:
    skill = _skill_spec(skill).id
    instructions = skill_instructions(skill, mode)

    local_contract = f"""
# Binding local ComfyUI output contract

This node performs one local inference pass using the selected Skill and the detected input mode
({mode.upper()}, target duration {duration:.2f} seconds).

Follow the selected Skill's output instructions and the user's request. There is no mandatory H3 field order,
mode-specific prefix, timestamp rule, audio section, language, Markdown style, or response shape. Return the
requested content as-is, including arbitrary prose, structured text, questions, code, or a fixed test string when
the selected Skill asks for it. Do not add a wrapper, explanation, or formatting requirement that the selected
Skill does not request.
""".strip()
    return instructions + "\n\n" + local_contract


def output_issues(
    text: str,
    mode: str,
    duration: float,
    require_complete_silence: bool = False,
) -> list[str]:
    """Return only a basic empty-output diagnostic; content shape is intentionally unrestricted."""
    del mode, duration, require_complete_silence
    return ["The output is empty"] if not text.strip() else []
