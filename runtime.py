from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


NODE_ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = NODE_ROOT / "runtime_config.json"
LLAMA_SEED_MODULUS = 0xFFFFFFFF


def normalize_llama_seed(seed: int) -> int:
    """Map ComfyUI's uint64 seed to llama.cpp's deterministic uint32 range."""
    return int(seed) % LLAMA_SEED_MODULUS


@dataclass(frozen=True)
class RuntimeSpec:
    executable: Path
    library_dirs: tuple[Path, ...]
    platform_name: str
    backend: str
    n_gpu_layers: str
    fit: bool
    fit_target_mib: int
    flash_attention: str


def _configured_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid {label} in {RUNTIME_CONFIG_PATH.name}")
    path = (root / Path(value)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} must remain inside the custom node directory: {path}"
        ) from error
    return path


def normalize_runtime_platform(system_name: str | None = None) -> str:
    value = (system_name or platform.system()).strip().casefold()
    names = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "macos",
        "macos": "macos",
    }
    if value not in names:
        raise RuntimeError(f"Unsupported operating system: {system_name or platform.system()}")
    return names[value]


def _installer_name(_platform_name: str) -> str:
    return "install_runtime.py"


def load_runtime_spec(
    root: Path = NODE_ROOT,
    system_name: str | None = None,
) -> RuntimeSpec:
    active_platform = normalize_runtime_platform(system_name)
    installer = _installer_name(active_platform)
    config_path = root / RUNTIME_CONFIG_PATH.name
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read {config_path}. Run {installer} --force."
            ) from error
        configured_platform = config.get("platform")
        if config.get("schema_version") != 1:
            raise RuntimeError(
                f"Unsupported runtime configuration in {config_path}. "
                f"Run {installer} --force."
            )
        if configured_platform != active_platform:
            raise RuntimeError(
                f"Runtime configuration is for {configured_platform}, but ComfyUI is "
                f"running on {active_platform}. Run {installer} --force."
            )
        executable = _configured_path(root, config.get("executable"), "executable")
        raw_library_dirs = config.get("library_dirs", [])
        if not isinstance(raw_library_dirs, list):
            raise RuntimeError(f"Invalid library_dirs in {config_path}")
        library_dirs = tuple(
            _configured_path(root, value, "library directory")
            for value in raw_library_dirs
        )
        options = config.get("runtime_options", {})
        if not isinstance(options, dict):
            raise RuntimeError(f"Invalid runtime_options in {config_path}")
        return RuntimeSpec(
            executable=executable,
            library_dirs=library_dirs or (executable.parent,),
            platform_name=active_platform,
            backend=str(config.get("backend", "unknown")),
            n_gpu_layers=str(options.get("n_gpu_layers", "auto")),
            fit=bool(options.get("fit", True)),
            fit_target_mib=int(options.get("fit_target_mib", 1536)),
            flash_attention=str(options.get("flash_attention", "auto")),
        )

    raise RuntimeError(
        f"llama.cpp runtime is not installed for {active_platform}. "
        f"Run {installer} from the custom node directory before starting ComfyUI."
    )


def build_runtime_environment(
    runtime_spec: RuntimeSpec,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    library_path = os.pathsep.join(str(path) for path in runtime_spec.library_dirs)
    environment["PATH"] = library_path + os.pathsep + environment.get("PATH", "")
    if runtime_spec.platform_name == "linux":
        environment["LD_LIBRARY_PATH"] = (
            library_path + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    elif runtime_spec.platform_name == "macos":
        environment["DYLD_LIBRARY_PATH"] = (
            library_path + os.pathsep + environment.get("DYLD_LIBRARY_PATH", "")
        )
    return environment


class LlamaServer:
    def __init__(
        self,
        model: Path,
        mmproj: Path,
        context_size: int = 32768,
        runtime_spec: RuntimeSpec | None = None,
    ):
        self.model = model
        self.mmproj = mmproj
        self.context_size = context_size
        self.runtime_spec = runtime_spec or load_runtime_spec()
        self.executable = self.runtime_spec.executable
        self.root = self.executable.parent
        self.backend = self.runtime_spec.backend
        self.port = self._free_port()
        self.process: subprocess.Popen | None = None
        self.log = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        try:
            self.start()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _log_tail(self) -> str:
        if self.log is None:
            return ""
        self.log.flush()
        self.log.seek(0)
        text = self.log.read().decode("utf-8", errors="replace")
        return text[-5000:]

    def start(self, timeout: float = 120.0) -> None:
        if not self.executable.is_file():
            installer = _installer_name(self.runtime_spec.platform_name)
            raise FileNotFoundError(
                f"llama.cpp runtime is missing: {self.executable}. "
                f"Run {installer} from the custom node directory."
            )
        self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        arguments = [
            str(self.executable),
            "--model",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--alias",
            "qwen3.8-27b",
            "--api-key",
            "comfyui-local",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            self.runtime_spec.n_gpu_layers,
            "--flash-attn",
            self.runtime_spec.flash_attention,
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--image-min-tokens",
            "1024",
            "--image-max-tokens",
            "1024",
            "--jinja",
            "--no-webui",
        ]
        if self.runtime_spec.fit:
            arguments.extend(
                [
                    "--fit",
                    "on",
                    "--fit-target",
                    str(self.runtime_spec.fit_target_mib),
                ]
            )
        env = build_runtime_environment(self.runtime_spec)
        process_options = {}
        if self.runtime_spec.platform_name == "windows":
            process_options["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        self.process = subprocess.Popen(
            arguments,
            cwd=self.root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            **process_options,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama.cpp exited while loading the model.\n{self._log_tail()}")
            try:
                self._request("GET", "/health", timeout=2.0)
                return
            except (OSError, RuntimeError):
                time.sleep(0.25)
        raise TimeoutError(
            f"llama.cpp did not become ready in {timeout:.0f} seconds.\n{self._log_tail()}"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def _request(self, method: str, path: str, payload=None, timeout: float = 900.0):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer comfyui-local",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {error.code}: {detail}") from error

    def chat(
        self,
        messages: list[dict[str, object]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        think_mode: bool,
        reasoning_effort: str,
    ) -> tuple[str, dict[str, int]]:
        payload = {
            "model": "qwen3.8-27b",
            "messages": messages,
            "seed": normalize_llama_seed(seed),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repeat_penalty": repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": think_mode,
                "preserve_thinking": False,
            },
        }
        if think_mode:
            payload["reasoning_effort"] = reasoning_effort
        response = self._request("POST", "/v1/chat/completions", payload)
        content = response["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise RuntimeError(
                "Qwen returned no final answer. Disable think mode or increase max_tokens so thinking does not consume the output budget."
            )
        return content.strip(), response.get("usage", {})


class LlamaServerManager:
    """Owns the one llama.cpp child process used by this custom node."""

    def __init__(self):
        self._lock = threading.RLock()
        self._server: LlamaServer | None = None
        self._key: tuple[Path, Path, int, Path, str, str] | None = None

    def acquire(
        self, model: Path, mmproj: Path, context_size: int = 32768
    ) -> tuple[LlamaServer, bool]:
        runtime_spec = load_runtime_spec()
        key = (
            model.resolve(),
            mmproj.resolve(),
            context_size,
            runtime_spec.executable.resolve(),
            runtime_spec.platform_name,
            runtime_spec.backend,
        )
        with self._lock:
            if self._server is not None and self._key == key and self._server.is_running:
                return self._server, False

            self.release()
            server = LlamaServer(model, mmproj, context_size, runtime_spec)
            try:
                server.start()
            except Exception:
                server.stop()
                raise
            self._server = server
            self._key = key
            return server, True

    def release(self) -> None:
        with self._lock:
            server = self._server
            self._server = None
            self._key = None
            if server is not None:
                server.stop()
