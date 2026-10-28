from __future__ import annotations

import time

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

from .common import (
    DEFAULT_MMPROJ,
    DEFAULT_MODEL,
    INFERENCE_LOCK,
    LOGGER,
    SERVER_MANAGER,
    ROUTING_MAX_TOKENS,
    autogrow_values as _autogrow_values,
    instruct_sampling_settings,
    media_content,
    model_options as _model_options,
    release_node_resources as _release_node_resources,
    resolve_model as _resolve_model,
    success as _success,
    failure as _failure,
    thinking_sampling_settings,
    usage_summary as _usage_summary,
    validate_images,
    video_details as _video_details,
)
from .skills import (
    SKILL_NAMES,
    detect_h3_mode,
    mode_router_prompt,
    output_issues,
    parse_mode_selection,
    parse_skill_selection,
    router_prompt,
    system_prompt,
)
from .general_node import QwenGeneralPrompt

REFERENCE_VIDEO_FPS = 24.0
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_VIDEO_SECONDS = 15.0


def _validate_reference_images(images: list) -> None:
    validate_images(images, MAX_REFERENCE_IMAGES)


def _reference_video_details(
    videos: list,
    sample_frames_per_second: int,
) -> list[tuple]:
    return _video_details(
        videos,
        REFERENCE_VIDEO_FPS,
        sample_frames_per_second,
        max_videos=MAX_REFERENCE_VIDEOS,
        max_video_seconds=MAX_REFERENCE_VIDEO_SECONDS,
        max_total_seconds=MAX_REFERENCE_VIDEO_SECONDS,
    )


def _picture_role(mode: str, picture_index: int) -> str:
    if mode == "i2va":
        return "the target video's first frame"
    if mode == "l2va":
        return "the target video's last frame"
    if mode == "fl2va":
        return "the target video's first frame" if picture_index == 1 else "the target video's last frame"
    return "a general visual reference whose role follows the user request"


def _sampling_settings(think_mode: bool) -> dict[str, float | int]:
    return thinking_sampling_settings() if think_mode else instruct_sampling_settings()


def _user_content(
    prompt: str,
    duration: float,
    images: list,
    video_details: list[tuple],
    mode: str,
) -> tuple[list[dict[str, object]], str]:
    return media_content(
        images,
        video_details,
        REFERENCE_VIDEO_FPS,
        image_role=lambda picture_index: _picture_role(mode, picture_index),
        include_prompt=(
            f"User request:\n{prompt}\n\nTarget duration: {duration:.2f} seconds."
        ),
    )


class QwenH3Prompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = _model_options(False)
        projectors = _model_options(True)
        return io.Schema(
            node_id="QwenH3PromptLocal",
            display_name="Qwen H3 Prompt (Local)",
            category="😺dzNodes/Qwen_H3_Prompt",
            description="Runs a local llama.cpp server with Qwen3.8 and a discovered H3 Skill from the bundled or custom Skill directories.",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="Describe the video or production task.",
                ),
                io.Combo.Input(
                    "skill",
                    options=["auto", *SKILL_NAMES],
                    default="auto",
                    tooltip="Auto routes across bundled and custom Skills, or choose one directly.",
                ),
                io.Float.Input(
                    "duration",
                    default=10.0,
                    min=1.0,
                    max=60.0,
                    step=0.1,
                    tooltip="Target H3 video duration in seconds.",
                ),
                io.Combo.Input(
                    "llm_model",
                    options=models,
                    default=DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                    tooltip="GGUF language model from ComfyUI/models/LLM/Qwen3.8.",
                ),
                io.Combo.Input(
                    "vision_model",
                    options=projectors,
                    default=DEFAULT_MMPROJ if DEFAULT_MMPROJ in projectors else projectors[0],
                    tooltip="GGUF multimodal projector from ComfyUI/models/LLM/Qwen3.8.",
                ),
                io.Boolean.Input(
                    "think_mode",
                    display_name="think_mode",
                    default=False,
                    tooltip="Off uses Qwen's official instruct settings. On enables thinking and uses Qwen's official thinking settings.",
                ),
                io.Combo.Input(
                    "reasoning_effort",
                    options=["low", "medium", "xhigh"],
                    default="medium",
                    tooltip="Applied only in thinking mode. Medium is the balanced RTX 3090 preset.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="ComfyUI seed. It is mapped deterministically to llama.cpp's 32-bit seed range.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=8192,
                    min=256,
                    max=8192,
                    step=128,
                    tooltip="Maximum generated tokens, including thinking when thinking mode is enabled.",
                ),
                io.Int.Input(
                    "video_sample_frames_per_sec",
                    default=2,
                    min=1,
                    max=8,
                    step=1,
                    advanced=True,
                    tooltip="Frames sampled from each second of every reference video. Frames from the same second are sent as one chronological contact sheet.",
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="force unload model",
                    default=True,
                    tooltip="On stops the node's llama.cpp server and clears its VRAM and system memory after each run. Errors always force an unload.",
                ),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_image",
                            tooltip="One reference image. State first-frame or last-frame intent explicitly in the prompt when needed.",
                        ),
                        prefix="reference_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "reference_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_video",
                            tooltip="Ordered video frames as an IMAGE batch, compatible with VHS Load Video.",
                        ),
                        prefix="reference_video_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.String.Output("h3_prompt"),
                io.String.Output("selected_skill"),
                io.String.Output("detected_mode"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        skill,
        duration,
        llm_model,
        vision_model,
        think_mode,
        reasoning_effort,
        seed,
        max_tokens,
        video_sample_frames_per_sec,
        force_unload_model,
        reference_images=None,
        reference_videos=None,
    ) -> io.NodeOutput:
        total_started = time.perf_counter()
        content = None
        messages = None
        repair_messages = None
        LOGGER.info(
            "[Qwen H3] Node execution started | skill=%s | think_mode=%s | seed=%d | force_unload_model=%s",
            skill,
            think_mode,
            seed,
            force_unload_model,
        )
        try:
            if not prompt.strip():
                raise ValueError("prompt must not be empty")

            stage_started = time.perf_counter()
            LOGGER.info("[Qwen H3] Preparing multimodal input")
            images = _autogrow_values(reference_images)
            videos = _autogrow_values(reference_videos)
            _validate_reference_images(images)
            video_details = _reference_video_details(
                videos,
                video_sample_frames_per_sec,
            )
            mode = detect_h3_mode(len(images), len(videos))
            model = _resolve_model(llm_model)
            projector = _resolve_model(vision_model)
            settings = _sampling_settings(think_mode)
            LOGGER.info(
                "[Qwen H3] Sampling configuration selected | mode=%s | temperature=%.2f | top_p=%.2f | top_k=%d | min_p=%.2f | presence_penalty=%.2f | repetition_penalty=%.2f",
                "thinking" if think_mode else "instruct",
                settings["temperature"],
                settings["top_p"],
                settings["top_k"],
                settings["min_p"],
                settings["presence_penalty"],
                settings["repetition_penalty"],
            )
            LOGGER.info(
                "[Qwen H3] Multimodal input parsed | images=%d | videos=%d | initial_mode=%s | elapsed %.2f s",
                len(images),
                len(videos),
                mode or "automatic",
                time.perf_counter() - stage_started,
            )

            with INFERENCE_LOCK:
                stage_started = time.perf_counter()
                LOGGER.info("[Qwen H3] Unloading ComfyUI models for Qwen")
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                LOGGER.info(
                    "[Qwen H3] ComfyUI model cleanup complete | elapsed %.2f s",
                    time.perf_counter() - stage_started,
                )

                stage_started = time.perf_counter()
                LOGGER.info(
                    "[Qwen H3] Model loading started | model=%s | mmproj=%s",
                    model.name,
                    projector.name,
                )
                server, loaded_new = SERVER_MANAGER.acquire(model, projector)
                LOGGER.info(
                    "[Qwen H3] Model loading complete | %s | platform=%s | backend=%s | port=%d | elapsed %.2f s",
                    "new model loaded" if loaded_new else "resident model reused",
                    server.runtime_spec.platform_name,
                    server.backend,
                    server.port,
                    time.perf_counter() - stage_started,
                )

                if mode is None:
                    stage_started = time.perf_counter()
                    LOGGER.info(
                        "[Qwen H3] Starting automatic H3 mode routing | images=%d",
                        len(images),
                    )
                    mode_selection, mode_usage = server.chat(
                        mode_router_prompt(prompt, len(images)),
                        seed=seed,
                        max_tokens=ROUTING_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        min_p=0.0,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        think_mode=False,
                        reasoning_effort="low",
                    )
                    mode = parse_mode_selection(mode_selection, len(images))
                    LOGGER.info(
                        "[Qwen H3] H3 mode routing complete | selected=%s | %s | elapsed %.2f s",
                        mode,
                        _usage_summary(mode_usage),
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] H3 mode routing result | selected=%s (deterministic from media counts)",
                        mode,
                    )

                stage_started = time.perf_counter()
                content, asset_summary = _user_content(
                    prompt,
                    duration,
                    images,
                    video_details,
                    mode,
                )
                LOGGER.info(
                    "[Qwen H3] Multimodal input ready | mode=%s | assets=%s | elapsed %.2f s",
                    mode,
                    asset_summary,
                    time.perf_counter() - stage_started,
                )

                selected = skill
                if selected == "auto":
                    stage_started = time.perf_counter()
                    LOGGER.info("[Qwen H3] Starting automatic Skill routing")
                    selection, routing_usage = server.chat(
                        router_prompt(prompt, mode, asset_summary),
                        seed=seed,
                        max_tokens=ROUTING_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        min_p=0.0,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        think_mode=False,
                        reasoning_effort="low",
                    )
                    selected = parse_skill_selection(selection)
                    LOGGER.info(
                        "[Qwen H3] Skill routing complete | selected=%s | %s | elapsed %.2f s",
                        selected,
                        _usage_summary(routing_usage),
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] Skill routing result | selected=%s (manual)",
                        selected,
                    )

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt(selected, mode, duration),
                    },
                    {"role": "user", "content": content},
                ]
                stage_started = time.perf_counter()
                LOGGER.info(
                    "[Qwen H3] Inference started | mode=%s | skill=%s | max_tokens=%d",
                    mode,
                    selected,
                    max_tokens,
                )
                result, inference_usage = server.chat(
                    messages,
                    seed=seed,
                    max_tokens=max_tokens,
                    think_mode=think_mode,
                    reasoning_effort=reasoning_effort,
                    **settings,
                )
                LOGGER.info(
                    "[Qwen H3] Inference complete | %s | elapsed %.2f s",
                    _usage_summary(inference_usage),
                    time.perf_counter() - stage_started,
                )

                LOGGER.info("[Qwen H3] Validating node output")
                issues = output_issues(
                    result,
                    mode,
                    duration,
                )
                if issues:
                    LOGGER.warning(
                        "[Qwen H3] Detected %d output issue(s); starting automatic repair: %s",
                        len(issues),
                        "; ".join(issues),
                    )
                    stage_started = time.perf_counter()
                    repair_messages = [
                        *messages,
                        {"role": "assistant", "content": result},
                        {
                            "role": "user",
                            "content": (
                                "Repair the output and return the complete requested content only. "
                                "Do not add a wrapper or impose an output schema that the selected Skill does not "
                                "request. Problems: " + "; ".join(issues)
                            ),
                        },
                    ]
                    result, repair_usage = server.chat(
                        repair_messages,
                        seed=seed,
                        max_tokens=max_tokens,
                        think_mode=think_mode,
                        reasoning_effort=reasoning_effort,
                        **settings,
                    )
                    LOGGER.info(
                        "[Qwen H3] Automatic repair inference complete | %s | elapsed %.2f s",
                        _usage_summary(repair_usage),
                        time.perf_counter() - stage_started,
                    )
                    remaining = output_issues(
                        result,
                        mode,
                        duration,
                    )
                    if remaining:
                        raise RuntimeError(
                            "Qwen output still fails the basic output contract: "
                            + "; ".join(remaining)
                        )
                LOGGER.info("[Qwen H3] Node output validation passed")

                content = None
                messages = None
                repair_messages = None
                if force_unload_model:
                    stage_started = time.perf_counter()
                    LOGGER.info("[Qwen H3] Force-releasing node model, VRAM, and system memory")
                    _release_node_resources()
                    LOGGER.info(
                        "[Qwen H3] Forced resource release complete | elapsed %.2f s",
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] Model unload disabled; the bundled llama.cpp model will remain resident for reuse"
                    )

            _success(
                f"[Qwen H3] Execution completed successfully | skill={selected} | mode={mode} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            return io.NodeOutput(result, selected, mode)
        except Exception as error:
            content = None
            messages = None
            repair_messages = None
            try:
                with INFERENCE_LOCK:
                    _release_node_resources()
            except Exception as cleanup_error:  # noqa: BLE001
                _failure(
                    f"[Qwen H3] Cleanup after failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _failure(
                f"[Qwen H3] Execution failed | {type(error).__name__}: {error} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            raise


class QwenH3PromptExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenH3Prompt, QwenGeneralPrompt]


async def comfy_entrypoint() -> QwenH3PromptExtension:
    return QwenH3PromptExtension()
