from __future__ import annotations

import atexit
import gc
import logging
import re
import threading
from pathlib import Path

import comfy.model_management
import folder_paths

from .media import (
    image_content,
    image_grid_content,
    sample_indices_per_second,
    text_content,
)
from .runtime import LlamaServerManager


MODEL_DIR = Path(folder_paths.models_dir) / "LLM" / "Qwen3.8"
DEFAULT_MODEL = "Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-F16.gguf"
NO_VISION_MODEL = "none"
CONTEXT_SIZE = 32768
ROUTING_MAX_TOKENS = 64
INFERENCE_LOCK = threading.Lock()
SERVER_MANAGER = LlamaServerManager()
LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt")
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

atexit.register(SERVER_MANAGER.release)


def success(message: str) -> None:
    LOGGER.info("%s%s%s", GREEN, message, RESET)


def failure(message: str) -> None:
    LOGGER.error("%s%s%s", RED, message, RESET)


def usage_summary(usage: dict[str, int]) -> str:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return "token usage unavailable"
    return (
        f"prompt {prompt_tokens or 0}, completion {completion_tokens or 0}, "
        f"total {total_tokens} tokens"
    )


def release_node_resources() -> None:
    SERVER_MANAGER.release()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)


def model_options(projector: bool) -> list[str]:
    files = sorted(path.name for path in MODEL_DIR.glob("*.gguf")) if MODEL_DIR.is_dir() else []
    if projector:
        options = [name for name in files if "mmproj" in name.lower()]
        fallback = DEFAULT_MMPROJ
    else:
        options = [name for name in files if "mmproj" not in name.lower()]
        fallback = DEFAULT_MODEL
    return options or [fallback]


def resolve_model(name: str, *, allow_missing: bool = False) -> Path | None:
    if allow_missing and name == NO_VISION_MODEL:
        return None
    root = MODEL_DIR.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file() or path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"Invalid or missing GGUF model in {root}: {name}")
    return path


def autogrow_values(inputs) -> list:
    if not inputs:
        return []

    def index(item):
        match = re.search(r"(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    return [value for _, value in sorted(inputs.items(), key=index) if value is not None]


def validate_images(images: list, max_images: int) -> None:
    if len(images) > max_images:
        raise ValueError(f"At most {max_images} images are supported; got {len(images)}.")
    for image_index, image in enumerate(images, 1):
        image_count = int(image.shape[0])
        if image_count != 1:
            raise ValueError(
                f"Image {image_index} must contain exactly one image; got a batch of {image_count}."
            )


def video_details(
    videos: list,
    fps: float,
    sample_frames_per_second: int,
    *,
    max_videos: int,
    max_video_seconds: float | None = None,
    max_total_seconds: float | None = None,
    max_sampled_frames_per_video: int | None = None,
) -> list[tuple]:
    if len(videos) > max_videos:
        raise ValueError(f"At most {max_videos} reference videos are supported; got {len(videos)}.")
    if fps <= 0:
        raise ValueError(f"video fps must be greater than 0; got {fps}.")
    details = []
    total_duration = 0.0
    for video_index, frames in enumerate(videos, 1):
        frame_count = int(frames.shape[0])
        if frame_count < 1:
            raise ValueError(
                f"Reference video {video_index} must contain at least 1 frame; got {frame_count}."
            )
        source_duration = frame_count / fps
        if max_video_seconds is not None and source_duration > max_video_seconds:
            raise ValueError(
                f"Reference video {video_index} must be at most {max_video_seconds:.2f} seconds at "
                f"{fps:.3f} fps; got {source_duration:.2f}s."
            )
        indices = sample_indices_per_second(frame_count, fps, sample_frames_per_second)
        sampled_frame_count = sum(len(group) for group in indices)
        if (
            max_sampled_frames_per_video is not None
            and sampled_frame_count > max_sampled_frames_per_video
        ):
            raise ValueError(
                f"Reference video {video_index} samples {sampled_frame_count} frames, which exceeds the "
                f"{max_sampled_frames_per_video} limit. Lower video_sample_frames_per_sec, raise video_fps, "
                f"or shorten the clip."
            )
        total_duration += source_duration
        details.append((frames, frame_count, source_duration, indices))
    if max_total_seconds is not None and total_duration > max_total_seconds:
        raise ValueError(
            f"Reference videos may total at most {max_total_seconds:.2f} seconds; "
            f"got {total_duration:.2f}s."
        )
    return details


def media_content(
    images: list,
    details: list[tuple],
    fps: float,
    *,
    image_role=None,
    include_prompt: str | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Assemble the multimodal user message shared by both nodes."""
    content: list[dict[str, object]] = []
    if include_prompt is not None:
        content.append(text_content(include_prompt))
    assets = []
    for picture_index, image in enumerate(images, 1):
        role = image_role(picture_index) if image_role else "a visual reference"
        content.append(text_content(f"<Picture {picture_index}>: connected as {role}."))
        content.append(image_content(image))
        assets.append(f"Picture {picture_index}={role}")

    for video_index, detail in enumerate(details, 1):
        frames, frame_count, source_duration, sample_groups = detail
        sampled_frame_count = sum(len(indices) for indices in sample_groups)
        content.append(
            text_content(
                f"<Video {video_index}>: {frame_count} ordered frames at {fps:.3f} fps "
                f"({source_duration:.2f} seconds), represented by {sampled_frame_count} sampled frames "
                f"grouped into {len(sample_groups)} chronological contact sheets."
            )
        )
        for second_index, indices in enumerate(sample_groups):
            timestamps = ", ".join(f"{frame_index / fps:.3f}s" for frame_index in indices)
            content.append(
                text_content(
                    f"<Video {video_index}> contact sheet {second_index + 1}/{len(sample_groups)}. "
                    f"Read cells in row-major chronological order at: {timestamps}."
                )
            )
            content.append(image_grid_content(frames, indices))
        assets.append(
            f"Video {video_index}={source_duration:.2f}s/{sampled_frame_count} sampled frames"
        )
    return content, ", ".join(assets) if assets else "none"


def instruct_sampling_settings() -> dict[str, float | int]:
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def thinking_sampling_settings() -> dict[str, float | int]:
    return {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }


def routing_sampling_settings() -> dict[str, float | int]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
