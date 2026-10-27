from __future__ import annotations

import time

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

from .common import (
    CONTEXT_SIZE,
    DEFAULT_MMPROJ,
    DEFAULT_MODEL,
    INFERENCE_LOCK,
    LOGGER,
    NO_VISION_MODEL,
    SERVER_MANAGER,
    autogrow_values,
    failure,
    instruct_sampling_settings,
    media_content,
    model_options,
    release_node_resources,
    resolve_model,
    routing_sampling_settings,
    success,
    thinking_sampling_settings,
    usage_summary,
    validate_images,
    video_details,
)
from .skills import (
    SKILL_NAMES,
    generic_router_prompt,
    generic_system_prompt,
    parse_skill_selection,
)


MAX_IMAGES = 12
MAX_VIDEOS = 3
MAX_SAMPLED_FRAMES_PER_VIDEO = 64
NO_SKILL = "none"
AUTO_SKILL = "auto"


def _skill_options() -> list[str]:
    return [NO_SKILL, AUTO_SKILL, *SKILL_NAMES] if SKILL_NAMES else [NO_SKILL, AUTO_SKILL]


def _sampling_settings(think_mode: bool) -> dict[str, float | int]:
    return thinking_sampling_settings() if think_mode else instruct_sampling_settings()


class QwenGeneralPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = model_options(False)
        projectors = model_options(True)
        skill_options = _skill_options()
        return io.Schema(
            node_id="QWENGeneralPromptLocal",
            display_name="QWEN General Prompt(Local)",
            category="😺dzNodes/Qwen_H3_Prompt",
            description=(
                "General-purpose local Qwen3.8 multimodal generation: images plus a text prompt, with an "
                "optional Skill from the bundled or custom Skill directories."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="Describe the connected images in detail.",
                    tooltip="The request sent to Qwen. Connected pictures follow it as visual input.",
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Extra standing instructions. Leave empty to rely on the prompt alone.",
                ),
                io.Combo.Input(
                    "skill",
                    options=skill_options,
                    default=NO_SKILL,
                    tooltip="None answers from the prompt only. Auto routes across bundled and custom Skills; pick one id to load it directly.",
                ),
                io.Boolean.Input(
                    "include_skill_references",
                    display_name="include skill references",
                    default=True,
                    advanced=True,
                    tooltip="Off loads only SKILL.md and skips the Skill's references/ files, which saves context.",
                ),
                io.Combo.Input(
                    "llm_model",
                    options=models,
                    default=DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                    tooltip="GGUF language model from ComfyUI/models/LLM/Qwen3.8.",
                ),
                io.Combo.Input(
                    "vision_model",
                    options=[NO_VISION_MODEL, *projectors],
                    default=DEFAULT_MMPROJ if DEFAULT_MMPROJ in projectors else NO_VISION_MODEL,
                    tooltip="GGUF multimodal projector, or none for text-only runs without an mmproj file.",
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
                    advanced=True,
                    tooltip="Applied only in thinking mode.",
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
                    max=16384,
                    step=128,
                    tooltip="Maximum generated tokens, including thinking when thinking mode is enabled.",
                ),
                io.Float.Input(
                    "video_fps",
                    default=24.0,
                    min=0.5,
                    max=120.0,
                    step=0.5,
                    advanced=True,
                    tooltip="Frame rate of every connected video frame batch. It sets the timestamps shown to the model.",
                ),
                io.Int.Input(
                    "video_sample_frames_per_sec",
                    default=2,
                    min=1,
                    max=8,
                    step=1,
                    advanced=True,
                    tooltip="Frames sampled from each second of every video. Frames from the same second are sent as one chronological contact sheet.",
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="force unload model",
                    default=True,
                    tooltip="On stops the node's llama.cpp server and clears its VRAM and system memory after each run. Errors always force an unload.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "image",
                            tooltip="One image to analyse. Each slot accepts a single IMAGE, not a batch.",
                        ),
                        prefix="image_",
                        min=0,
                        max=MAX_IMAGES,
                    ),
                ),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "video",
                            tooltip="Ordered video frames as an IMAGE batch, compatible with VHS Load Video.",
                        ),
                        prefix="video_",
                        min=0,
                        max=MAX_VIDEOS,
                    ),
                ),
            ],
            outputs=[
                io.String.Output("text"),
                io.String.Output("selected_skill"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        system_prompt,
        skill,
        include_skill_references,
        llm_model,
        vision_model,
        think_mode,
        reasoning_effort,
        seed,
        max_tokens,
        video_fps,
        video_sample_frames_per_sec,
        force_unload_model,
        images=None,
        videos=None,
    ) -> io.NodeOutput:
        total_started = time.perf_counter()
        content = None
        messages = None
        LOGGER.info(
            "[QWEN General] Node execution started | skill=%s | think_mode=%s | seed=%d | force_unload_model=%s",
            skill,
            think_mode,
            seed,
            force_unload_model,
        )
        try:
            if not prompt.strip():
                raise ValueError("prompt must not be empty")
            if skill not in _skill_options():
                raise ValueError(f"Unknown skill: {skill}")

            stage_started = time.perf_counter()
            LOGGER.info("[QWEN General] Preparing multimodal input")
            image_list = autogrow_values(images)
            video_list = autogrow_values(videos)
            validate_images(image_list, MAX_IMAGES)
            details = video_details(
                video_list,
                video_fps,
                video_sample_frames_per_sec,
                max_videos=MAX_VIDEOS,
                max_sampled_frames_per_video=MAX_SAMPLED_FRAMES_PER_VIDEO,
            )
            model = resolve_model(llm_model)
            projector = resolve_model(vision_model, allow_missing=True)
            if projector is None and (image_list or video_list):
                raise ValueError(
                    "vision_model is none but images or videos are connected; select an mmproj file to read them."
                )
            settings = _sampling_settings(think_mode)
            LOGGER.info(
                "[QWEN General] Sampling configuration selected | mode=%s | temperature=%.2f | top_p=%.2f | top_k=%d | min_p=%.2f | presence_penalty=%.2f | repetition_penalty=%.2f",
                "thinking" if think_mode else "instruct",
                settings["temperature"],
                settings["top_p"],
                settings["top_k"],
                settings["min_p"],
                settings["presence_penalty"],
                settings["repetition_penalty"],
            )
            LOGGER.info(
                "[QWEN General] Multimodal input parsed | images=%d | videos=%d | elapsed %.2f s",
                len(image_list),
                len(video_list),
                time.perf_counter() - stage_started,
            )

            with INFERENCE_LOCK:
                stage_started = time.perf_counter()
                LOGGER.info("[QWEN General] Unloading ComfyUI models for Qwen")
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                LOGGER.info(
                    "[QWEN General] ComfyUI model cleanup complete | elapsed %.2f s",
                    time.perf_counter() - stage_started,
                )

                stage_started = time.perf_counter()
                LOGGER.info(
                    "[QWEN General] Model loading started | model=%s | mmproj=%s",
                    model.name,
                    projector.name if projector else NO_VISION_MODEL,
                )
                server, loaded_new = SERVER_MANAGER.acquire(model, projector, CONTEXT_SIZE)
                LOGGER.info(
                    "[QWEN General] Model loading complete | %s | platform=%s | backend=%s | port=%d | elapsed %.2f s",
                    "new model loaded" if loaded_new else "resident model reused",
                    server.runtime_spec.platform_name,
                    server.backend,
                    server.port,
                    time.perf_counter() - stage_started,
                )

                content, asset_summary = media_content(
                    image_list,
                    details,
                    video_fps,
                    include_prompt=prompt.strip(),
                )
                stage_started = time.perf_counter()
                LOGGER.info(
                    "[QWEN General] Multimodal input ready | assets=%s | elapsed %.2f s",
                    asset_summary,
                    time.perf_counter() - stage_started,
                )

                selected = skill
                if skill == AUTO_SKILL:
                    if not SKILL_NAMES:
                        raise RuntimeError(
                            "No Skill was discovered, so auto routing is unavailable. Add a SKILL.md under "
                            "skills/ or custom_skills/, or set skill to none."
                        )
                    stage_started = time.perf_counter()
                    LOGGER.info("[QWEN General] Starting automatic Skill routing")
                    selection, routing_usage = server.chat(
                        generic_router_prompt(prompt, asset_summary),
                        seed=seed,
                        max_tokens=48,
                        think_mode=False,
                        reasoning_effort="low",
                        **routing_sampling_settings(),
                    )
                    selected = parse_skill_selection(selection)
                    if not selected:
                        raise RuntimeError("Automatic Skill routing did not resolve to a known Skill.")
                    LOGGER.info(
                        "[QWEN General] Skill routing complete | selected=%s | %s | elapsed %.2f s",
                        selected,
                        usage_summary(routing_usage),
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[QWEN General] Skill selection | selected=%s (%s)",
                        selected,
                        "manual" if skill != NO_SKILL else "no Skill",
                    )

                messages = [
                    {
                        "role": "system",
                        "content": generic_system_prompt(
                            selected if selected != NO_SKILL else None,
                            system_prompt,
                            include_references=include_skill_references,
                        ),
                    },
                    {"role": "user", "content": content},
                ]
                stage_started = time.perf_counter()
                LOGGER.info(
                    "[QWEN General] Inference started | skill=%s | max_tokens=%d",
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
                    "[QWEN General] Inference complete | %s | elapsed %.2f s",
                    usage_summary(inference_usage),
                    time.perf_counter() - stage_started,
                )

                if not result.strip():
                    raise RuntimeError("Qwen returned an empty answer.")

                content = None
                messages = None
                if force_unload_model:
                    stage_started = time.perf_counter()
                    LOGGER.info("[QWEN General] Force-releasing node model, VRAM, and system memory")
                    release_node_resources()
                    LOGGER.info(
                        "[QWEN General] Forced resource release complete | elapsed %.2f s",
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[QWEN General] Model unload disabled; the bundled llama.cpp model will remain resident for reuse"
                    )

            success(
                f"[QWEN General] Execution completed successfully | skill={selected} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            return io.NodeOutput(result, selected)
        except Exception as error:
            content = None
            messages = None
            try:
                with INFERENCE_LOCK:
                    release_node_resources()
            except Exception as cleanup_error:  # noqa: BLE001
                failure(
                    f"[QWEN General] Cleanup after failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            failure(
                f"[QWEN General] Execution failed | {type(error).__name__}: {error} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            raise


__all__ = ["QwenGeneralPrompt"]
