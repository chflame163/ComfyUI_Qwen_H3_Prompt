from __future__ import annotations

import re
from pathlib import Path

SKILL_ROOT = Path(__file__).with_name("skills")

SKILL_NAMES = (
    "3d-animation-short-generator",
    "brand-promo-video-generator",
    "co-op-game-intro-generator",
    "h3-prompt-writing",
    "handdrawn-live-video-generator",
    "minimalist-product-ad-generator",
    "music-video-subtitle-generator",
    "paper-collage-explainer-generator",
    "papercraft-stop-motion-explainer",
)

SKILL_DESCRIPTIONS = {
    "3d-animation-short-generator": "Complete story-first stylized 3D animated shorts with characters, environments, shots, continuity, audio, and review.",
    "brand-promo-video-generator": "Promotional shorts for brands, products, apps, shops, websites, and launches using verified assets and claims.",
    "co-op-game-intro-generator": "Two-player co-op game menu or opening animation with player cards, characters, and menu interaction.",
    "h3-prompt-writing": "Direct MiniMax H3 prompt writing for T2VA, I2VA, FL2VA, L2VA, and Ref2VA.",
    "handdrawn-live-video-generator": "Surreal live-action scenes blended with rough glowing hand-drawn animation, contact, morphing, and chase motion.",
    "minimalist-product-ad-generator": "Minimalist premium product advertising shorts for e-commerce and product launches.",
    "music-video-subtitle-generator": "Music videos and emotional shorts with lyric timing, spatial typography, subtitles, beats, and shot continuity.",
    "paper-collage-explainer-generator": "Tactile halftone paper-collage explainers for narration, concepts, opinions, and social video.",
    "papercraft-stop-motion-explainer": "Handmade papercraft, cut-paper, pop-up-book, diorama, and stop-motion educational explainers.",
}

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


def router_prompt(user_prompt: str, mode: str, asset_summary: str) -> list[dict[str, str]]:
    catalog = "\n".join(f"- {name}: {SKILL_DESCRIPTIONS[name]}" for name in SKILL_NAMES)
    system = (
        "Select exactly one MiniMax H3 skill for the request. Return only its exact skill id, "
        "with no punctuation or explanation. Prefer h3-prompt-writing for direct prompt rewriting "
        "when no specialized production workflow clearly applies.\n\n" + catalog
    )
    user = f"Detected H3 mode: {mode}\nConnected assets: {asset_summary}\nRequest:\n{user_prompt}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_skill_selection(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in SKILL_NAMES:
        return normalized
    matches = [name for name in SKILL_NAMES if name in normalized]
    if len(matches) == 1:
        return matches[0]
    return "h3-prompt-writing"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def skill_instructions(skill: str, mode: str, max_chars: int = 72000) -> str:
    if skill not in SKILL_NAMES:
        raise ValueError(f"Unknown skill: {skill}")
    root = (SKILL_ROOT / skill).resolve()
    if root.parent != SKILL_ROOT.resolve() or not root.is_dir():
        raise FileNotFoundError(f"Bundled skill is missing: {skill}")

    parts = [f"# Official MiniMax Skill: {skill}\n\n{_read(root / 'SKILL.md')}"]
    if skill == "h3-prompt-writing":
        guide = "ref-en.txt" if mode == "ref2va" else "base-en.txt"
        parts.append(f"# Required reference: {guide}\n\n{_read(root / 'references' / guide)}")
    else:
        used = len(parts[0])
        for path in (
            sorted((root / "references").glob("**/*")) if (root / "references").is_dir() else ()
        ):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = _read(path)
            addition = f"# Official reference: {path.relative_to(root).as_posix()}\n\n{text}"
            if used + len(addition) > max_chars:
                break
            parts.append(addition)
            used += len(addition)
    return "\n\n".join(parts)


def system_prompt(skill: str, mode: str, duration: float) -> str:
    instructions = skill_instructions(skill, mode)
    if skill != "h3-prompt-writing":
        guide = "ref-en.txt" if mode == "ref2va" else "base-en.txt"
        instructions += f"\n\n# Required final H3 prompt format: {guide}\n\n" + _read(
            SKILL_ROOT / "h3-prompt-writing" / "references" / guide
        )

    local_contract = f"""
# Binding local ComfyUI output contract (overrides conflicting workflow instructions above)

This node is a single-pass MiniMax H3 video-prompt generator, not an interactive MiniMax Hub workflow.
Use the selected Skill only as creative and domain guidance, then return exactly one complete final H3 prompt
in the required H3 format. The detected input mode is {mode.upper()} and the exact target duration is
{duration:.2f} seconds.

Never ask the user a question. Never request clarification, confirmation, approval, assets, choices, or a reply.
Never output an intake form, pre-production package, source summary, option list, confirmation gate, approval gate,
required-clarifications section, next steps, generation offer, or instructions such as "confirm", "go ahead", or
"proceed". Ignore every official Skill instruction that says to ask, pause, wait, present a card, obtain approval,
or continue after confirmation. Do not mention those omitted workflow phases in the answer.

When optional information is absent, silently omit the corresponding optional element. In particular, do not add
on-screen copy, slogans, logos, brand claims, dialogue, narration, lyrics, or product facts unless the user supplied
them or they are unambiguously visible in a connected reference. Never invent a factual claim. When an unspecified
creative detail is necessary to make the video executable, choose one conservative value that is compatible with
the request and visible references; do not present alternatives or expose the assumption.

Unless the user explicitly requests complete silence, create an original audio design from the user request, visible
image and sampled-video content, and the exact target duration. Fill `overall_soundscape` with time-compatible ambience
and physical action sounds. Fill `non_diegetic_music` with a concise instrumental score describing mood, suitable
instruments, pacing, development, synchronization with major visual beats, and the ending. Do not claim to hear,
analyze, copy, or preserve audio from a connected video: this node receives video frames without their audio tracks.
Do not invent vocals, lyrics, spoken words, or narration. Respect an explicit request for no background music by using
`non_diegetic_music: N/A` while retaining appropriate diegetic sound. If the user explicitly requests complete silence,
output exactly `overall_soundscape: N/A` and `non_diegetic_music: N/A`, and include no audible events elsewhere.

Return the final prompt only, with no Markdown fences, title, analysis, preface, epilogue, notes, caveats, checklist,
or follow-up text. Preserve user-provided dialogue, lyrics, and visible text exactly. Treat no dialogue and no
background music as independent constraints; retain plausible diegetic ambience and action sounds unless the user
explicitly requests complete silence.
""".strip()
    return instructions + "\n\n" + local_contract


def output_issues(
    text: str,
    mode: str,
    duration: float,
    require_complete_silence: bool = False,
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["The output is empty"]
    issues: list[str] = []
    if "```" in stripped:
        issues.append("Remove Markdown code fences")

    interaction_scope = re.sub(r"<d>.*?</d>", "", stripped, flags=re.DOTALL)
    if any(pattern.search(interaction_scope) for pattern in INTERACTIVE_OUTPUT_PATTERNS):
        issues.append(
            "Remove all questions, clarifications, confirmations, approval gates, and next-step requests"
        )

    fields = REF_FIELDS if mode == "ref2va" else BASE_FIELDS
    positions = [stripped.find(field) for field in fields]
    missing = [field for field, position in zip(fields, positions, strict=True) if position < 0]
    if missing:
        issues.append(f"Missing required fields: {', '.join(missing)}")
    elif positions != sorted(positions):
        issues.append("Required fields are not in the official order")

    if mode == "t2va" and not stripped.startswith(BASE_FIELDS[0]):
        issues.append("T2VA must begin directly with integrated_multimodal_description")
    if mode == "ref2va" and not stripped.startswith(REF_FIELDS[0]):
        issues.append("REF2VA must begin directly with subject_definitions")

    if mode == "i2va" and not stripped.startswith(
        "For the target video, at 0.00 seconds into the target video,"
    ):
        issues.append("I2VA must begin with the official first-frame instruction")
    if mode in {"fl2va", "l2va"} and not stripped.startswith(
        "How the reference pictures align with the target video"
    ):
        issues.append(f"{mode.upper()} must begin with the official alignment instruction")

    if require_complete_silence:
        soundscape_is_silent = re.search(
            r"(?m)^overall_soundscape:\s*N/A\s*$", stripped
        )
        music_is_silent = re.search(
            r"(?m)^non_diegetic_music:\s*N/A\s*$", stripped
        )
        if not soundscape_is_silent or not music_is_silent:
            issues.append(
                "Complete silence was requested: set both overall_soundscape and non_diegetic_music exactly to N/A"
            )

    previous = -1.0
    for match in TIMESTAMP_RE.finditer(stripped):
        timestamp = int(match.group(1)) * 60 + int(match.group(2)) + int(match.group(3)) / 1000
        if timestamp <= previous:
            issues.append("Shot timestamps must be strictly increasing")
            break
        if timestamp >= duration:
            issues.append(f"Shot timestamp {timestamp:.3f}s must be earlier than {duration:.2f}s")
            break
        previous = timestamp
    return issues
