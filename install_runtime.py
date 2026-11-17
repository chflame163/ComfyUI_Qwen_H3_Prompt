from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
USER_AGENT = "ComfyUI-Qwen-H3-Prompt-runtime-installer/1.0"
ROOT = Path(__file__).resolve().parent
RELEASE_TAG = "b12464"
RELEASE_COMMIT_FULL = "d3c3e242c16b8cbf51400fad410cd69b22d64f35"
RELEASE_COMMIT = RELEASE_COMMIT_FULL
RELEASE_BASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE_TAG}"
)
SOURCE_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
SOURCE_CONFIG_VALUE = "official ggml-org/llama.cpp source built locally with CUDA"


@dataclass(frozen=True)
class Asset:
    name: str
    sha256: str
    base_url: str

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.name}"


@dataclass(frozen=True)
class RuntimePackage:
    backend: str
    description: str
    assets: tuple[Asset, ...]
    device_markers: tuple[str, ...]
    cuda_version: tuple[int, int] | None = None


@dataclass(frozen=True)
class InstallerContext:
    root: Path
    platform_name: str
    architecture: str
    release_tag: str
    release_commit: str
    executable_name: str = "llama-server"

    @property
    def runtime_root(self) -> Path:
        return self.root / "runtime"

    @property
    def config_path(self) -> Path:
        return self.root / "runtime_config.json"


@dataclass(frozen=True)
class HardwareInfo:
    architecture: str
    gpu_names: tuple[str, ...]
    vendors: tuple[str, ...] = ()
    cuda_driver_version: tuple[int, int] | None = None
    cuda_toolkit_version: tuple[int, int] | None = None

    @property
    def cuda_selection_version(self) -> tuple[int, int] | None:
        # Release archives bundle the CUDA runtime. Driver compatibility is what
        # determines whether they can run; nvcc is only a fallback signal.
        return self.cuda_driver_version or self.cuda_toolkit_version


@dataclass(frozen=True)
class CudaBuildTools:
    git: str
    cmake: str
    cuda_compiler: Path
    cuda_root: Path
    cuda_library_dirs: tuple[Path, ...]
    c_compiler: str
    cxx_compiler: str


def _asset(name: str, sha256: str) -> Asset:
    return Asset(name=name, sha256=sha256, base_url=RELEASE_BASE_URL)


# Verified llama.cpp release catalogs for every supported platform.
WINDOWS_PACKAGES: dict[str, dict[str, RuntimePackage]] = {
    "x64": {
        "cuda12": RuntimePackage(
            backend="cuda12",
            description="NVIDIA CUDA 12.4",
            assets=(
                _asset(
                    "llama-b12464-bin-win-cuda-12.4-x64.zip",
                    "d3790fd98ce4765b645cb233aab57d359ebf182d5b6c9bbf6d03d9b8794dda85",
                ),
                _asset(
                    "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(12, 4),
        ),
        "cuda13": RuntimePackage(
            backend="cuda13",
            description="NVIDIA CUDA 13.5",
            assets=(
                _asset(
                    "llama-b12464-bin-win-cuda-13.5-x64.zip",
                    "a740899a7115dc8db1640e4ee74f9e494e419f2ad1ebed2805b5486d9cfbfc2c",
                ),
                _asset(
                    "cudart-llama-bin-win-cuda-13.5-x64.zip",
                    "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(13, 5),
        ),
        "rocm": RuntimePackage(
            backend="rocm",
            description="AMD ROCm 10.0",
            assets=(
                _asset(
                    "llama-b12464-bin-win-rocm-10.0-x64.zip",
                    "791e0e2e477a676f936518b4b753bdb1e2d67d1246aa6c91c9e9139b67eac4bb",
                ),
            ),
            device_markers=("ROCM", "HIP"),
        ),
        "sycl": RuntimePackage(
            backend="sycl",
            description="Intel SYCL",
            assets=(
                _asset(
                    "llama-b12464-bin-win-sycl-x64.zip",
                    "ade8662ee14d8af2468c8318b4da6513a2471b589c67dfacf2ea42a9e6e934f2",
                ),
            ),
            device_markers=("SYCL", "LEVEL-ZERO", "INTEL"),
        ),
        "vulkan": RuntimePackage(
            backend="vulkan",
            description="Vulkan (NVIDIA, AMD, or Intel)",
            assets=(
                _asset(
                    "llama-b12464-bin-win-vulkan-x64.zip",
                    "839455639ced0f6ad50e971d37c89050e0e8b952c12e1c62a78823940fb93073",
                ),
            ),
            device_markers=("VULKAN",),
        ),
        "cpu": RuntimePackage(
            backend="cpu",
            description="CPU x64",
            assets=(
                _asset(
                    "llama-b12464-bin-win-cpu-x64.zip",
                    "2bc4bfd76278c57cc528d150a7ffc598d2ba3c4ec0c6f2ec1963fc492217442d",
                ),
            ),
            device_markers=(),
        ),
    },
    "arm64": {
        "cuda13": RuntimePackage(
            backend="cuda13",
            description="NVIDIA CUDA 13.5 ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-win-cuda-13.5-arm64.zip",
                    "7f1f022adae807bcf46c7a422a3ccc0ab4038af33244f88eeab85c50b58ad11e",
                ),
                _asset(
                    "cudart-llama-bin-win-cuda-13.5-arm64.zip",
                    "f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(13, 5),
        ),
        "opencl": RuntimePackage(
            backend="opencl",
            description="Qualcomm Adreno OpenCL ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-win-opencl-adreno-arm64.zip",
                    "7f4f4d1fffd1c23f1f6fa9cc6113583078d2b9e7cfefccb41471fb49552d1bcf",
                ),
            ),
            device_markers=("OPENCL", "ADRENO"),
        ),
        "cpu": RuntimePackage(
            backend="cpu",
            description="CPU ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-win-cpu-arm64.zip",
                    "7d69e04db6cfaa116e94c61d27bd12e6ca594b1397460af8268575475427ff2f",
                ),
            ),
            device_markers=(),
        ),
    },
}


LINUX_PACKAGES: dict[str, dict[str, RuntimePackage]] = {
    "x64": {
        "cuda12": RuntimePackage(
            backend="cuda12",
            description="NVIDIA CUDA 12.8 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-cuda-12.8-x64.tar.gz",
                    "c4f0ef9755a180e826eefe5e544826543f684c424ea88b979e8e1a11e248b043",
                ),
                _asset(
                    "cudart-llama-b12464-bin-ubuntu-cuda-12.8-x64.tar.gz",
                    "5e33af47a127cc537e075ab4e811280c9ef20c048157717778e3ca547e01cf9f",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(12, 8),
        ),
        "cuda13": RuntimePackage(
            backend="cuda13",
            description="NVIDIA CUDA 13.5 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-cuda-13.5-x64.tar.gz",
                    "e7656d43e84ee45e493e5aa0714d50a34ad698d9e2dac116b79233678e2b7ebe",
                ),
                _asset(
                    "cudart-llama-b12464-bin-ubuntu-cuda-13.5-x64.tar.gz",
                    "f9cb16c7f99055d823df09c478e5af04d131e928a70ac4af8724a121b0e773ff",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(13, 5),
        ),
        "rocm": RuntimePackage(
            backend="rocm",
            description="AMD ROCm 10.0 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-rocm-10.0-x64.tar.gz",
                    "e344afe9e78896cd5eb0698815d79e06faa68aa58569e1ea81620146f5b13d7a",
                ),
            ),
            device_markers=("ROCM", "HIP"),
        ),
        "vulkan": RuntimePackage(
            backend="vulkan",
            description="Vulkan x64 (NVIDIA, AMD, or Intel)",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-vulkan-x64.tar.gz",
                    "78fb7eefb8f022e0f181083e7f868917efc184b38cd9282ffc450c950bbae018",
                ),
            ),
            device_markers=("VULKAN",),
        ),
        "sycl": RuntimePackage(
            backend="sycl",
            description="Intel SYCL FP16 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-sycl-fp16-x64.tar.gz",
                    "413d42175a46a23b65f6ed60a2ccb735963d69de7c036c3855b14371f6eadb99",
                ),
            ),
            device_markers=("SYCL", "LEVEL-ZERO", "INTEL"),
        ),
        "sycl-fp32": RuntimePackage(
            backend="sycl-fp32",
            description="Intel SYCL FP32 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-sycl-fp32-x64.tar.gz",
                    "92e543d8cf29c49082e15bbb0b7c547e3264bf3c1edd4788c3050fc229c3e51f",
                ),
            ),
            device_markers=("SYCL", "LEVEL-ZERO", "INTEL"),
        ),
        "openvino": RuntimePackage(
            backend="openvino",
            description="Intel OpenVINO 2026.4.1 x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
                    "88f3481260df7a58d53e4380d529ceb8aee3ec228804b59fcd307152ea31b970",
                ),
            ),
            device_markers=("OPENVINO", "INTEL"),
        ),
        "cpu": RuntimePackage(
            backend="cpu",
            description="CPU x64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-x64.tar.gz",
                    "7dd1899af7f0ddda42920d383b7dcda7097c1e5e45fcef582b68ea66badb46de",
                ),
            ),
            device_markers=(),
        ),
    },
    "arm64": {
        "cuda13": RuntimePackage(
            backend="cuda13",
            description="NVIDIA CUDA 13.5 ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                    "fbf58f5247cc4b47e0a3049c480619f242470bf2109249de08a0ab1446e54411",
                ),
                _asset(
                    "cudart-llama-b12464-bin-ubuntu-cuda-13.5-arm64.tar.gz",
                    "90c736433e49424252c0ba4f84a82a0c44ec10994fd62dbae8c5ae77b7968ff3",
                ),
            ),
            device_markers=("CUDA",),
            cuda_version=(13, 5),
        ),
        "vulkan": RuntimePackage(
            backend="vulkan",
            description="Vulkan ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-vulkan-arm64.tar.gz",
                    "ca79fc30abebd109b08899a59d65753293827605cf1e9feb6e323899b04d719e",
                ),
            ),
            device_markers=("VULKAN",),
        ),
        "cpu": RuntimePackage(
            backend="cpu",
            description="CPU ARM64",
            assets=(
                _asset(
                    "llama-b12464-bin-ubuntu-arm64.tar.gz",
                    "cdc29168b4a214c5e4c796c8218694b7ece9317ddf1e1641b4325b0f0dd6eaa3",
                ),
            ),
            device_markers=(),
        ),
    },
}


LINUX_CUDA_PACKAGES: dict[str, RuntimePackage] = {
    architecture: RuntimePackage(
        backend="cuda",
        description=f"CUDA {architecture} (locally built from llama.cpp {RELEASE_TAG})",
        assets=(),
        device_markers=("CUDA",),
    )
    for architecture in LINUX_PACKAGES
}


def _macos_packages(
    asset_name: str,
    sha256: str,
    architecture: str,
) -> dict[str, RuntimePackage]:
    asset = _asset(asset_name, sha256)
    return {
        "metal": RuntimePackage(
            backend="metal",
            description=f"Metal {architecture}",
            assets=(asset,),
            device_markers=("METAL",),
        ),
        "cpu": RuntimePackage(
            backend="cpu",
            description=f"CPU {architecture}",
            assets=(asset,),
            device_markers=(),
        ),
    }


MACOS_PACKAGES: dict[str, dict[str, RuntimePackage]] = {
    "arm64": _macos_packages(
        "llama-b12464-bin-macos-arm64.tar.gz",
        "95cb3cf68c5534bebe598c4f95576fcca28444627de2373293512759e3ffa925",
        "ARM64",
    ),
    "x64": _macos_packages(
        "llama-b12464-bin-macos-x64.tar.gz",
        "7fea952a34e74a63f7ed6fbc5469be3f50374a8906e849d30459fc57ba01ca56",
        "x64",
    ),
}


PLATFORM_PACKAGES = {
    "windows": WINDOWS_PACKAGES,
    "linux": LINUX_PACKAGES,
    "macos": MACOS_PACKAGES,
}


# Shared download, extraction, validation, and atomic installation primitives.
def print_success(message: str) -> None:
    _print_colored(message, GREEN)


def print_error(message: str) -> None:
    _print_colored(message, RED, stream=sys.stderr)


def _color_supported(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None or not stream.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        handle_id = -11 if stream is sys.stdout else -12
        handle = ctypes.windll.kernel32.GetStdHandle(handle_id)
        mode = ctypes.c_uint32()
        if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        )
    except (AttributeError, OSError):
        return False


def _print_colored(message: str, color: str, *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    if _color_supported(stream):
        print(f"{color}{message}{RESET}", file=stream)
    else:
        print(message, file=stream)


def run_probe(arguments: list[str], timeout: float = 12.0) -> str:
    process_options = {}
    if os.name == "nt":
        process_options["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **process_options,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()


def unique_lines(text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\ufeff")
        key = line.casefold()
        if line and key not in seen:
            seen.add(key)
            result.append(line)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_progress(downloaded: int, total: int) -> str:
    downloaded_mib = downloaded / (1024 * 1024)
    if total:
        return (
            f"[INFO] Download progress: {downloaded / total * 100:5.1f}% "
            f"({downloaded_mib:.1f}/{total / (1024 * 1024):.1f} MiB)"
        )
    return f"[INFO] Download progress: {downloaded_mib:.1f} MiB"


def download_asset(asset: Asset, cache_dir: Path, offline: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset.name
    if destination.is_file():
        print(f"[INFO] Verifying cached asset: {asset.name}")
        if sha256_file(destination) == asset.sha256:
            print(f"[OK] Cached asset verified: {asset.name}")
            return destination
        print(f"[WARN] Removing cached asset with an invalid checksum: {asset.name}")
        destination.unlink()

    if offline:
        raise RuntimeError(f"Offline cache does not contain a valid {asset.name}")

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(asset.url, headers={"User-Agent": USER_AGENT})
    print(f"[INFO] Downloading {asset.name}")
    started = time.monotonic()
    downloaded = 0
    total = 0
    interactive_progress = sys.stdout.isatty()
    progress_width = 0
    progress_visible = False
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            last_update = 0.0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if interactive_progress and now - last_update >= 0.2:
                    line = _download_progress(downloaded, total)
                    progress_width = max(progress_width, len(line))
                    sys.stdout.write("\r" + line.ljust(progress_width))
                    sys.stdout.flush()
                    progress_visible = True
                    last_update = now
    except (OSError, urllib.error.URLError) as error:
        if progress_visible:
            sys.stdout.write("\n")
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"Download failed for {asset.name}: {error}") from error

    final_progress = _download_progress(downloaded, total)
    if interactive_progress:
        progress_width = max(progress_width, len(final_progress))
        sys.stdout.write("\r" + final_progress.ljust(progress_width) + "\n")
        sys.stdout.flush()
    else:
        print(final_progress)

    actual = sha256_file(partial)
    if actual != asset.sha256:
        partial.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
        )
    os.replace(partial, destination)
    print(
        f"[OK] Downloaded and verified {asset.name} "
        f"in {time.monotonic() - started:.1f} s"
    )
    return destination


def _safe_destination(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or (pure_path.parts and ":" in pure_path.parts[0])
    ):
        raise RuntimeError(f"Unsafe path in archive: {member_name}")
    destination = (root / Path(*pure_path.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Unsafe path traversal in archive: {member_name}") from error
    return destination


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = _safe_destination(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _validate_tar_member(root: Path, member: tarfile.TarInfo) -> None:
    target = _safe_destination(root, member.name)
    if member.ischr() or member.isblk() or member.isfifo():
        raise RuntimeError(f"Unsupported special file in archive: {member.name}")
    if member.issym():
        link_target = (target.parent / member.linkname).resolve()
    elif member.islnk():
        link_target = _safe_destination(root, member.linkname)
    else:
        return
    try:
        link_target.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Unsafe link in archive: {member.name}") from error


def _extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(destination, member)
        try:
            archive.extractall(
                destination,
                members=members,
                filter="fully_trusted",
            )
        except TypeError:
            archive.extractall(destination, members=members)


def extract_archive_safely(archive_path: Path, destination: Path) -> None:
    lower_name = archive_path.name.casefold()
    if lower_name.endswith(".zip"):
        _extract_zip(archive_path, destination)
        return
    if lower_name.endswith((".tar.gz", ".tgz", ".tar.xz", ".tar")):
        _extract_tar(archive_path, destination)
        return
    raise RuntimeError(f"Unsupported runtime archive format: {archive_path.name}")


def find_server(root: Path, executable_name: str) -> Path:
    candidates = sorted(
        (path for path in root.rglob(executable_name) if path.is_file()),
        key=lambda path: (len(path.relative_to(root).parts), str(path).casefold()),
    )
    if not candidates:
        raise RuntimeError(f"The runtime archive does not contain {executable_name}")
    return candidates[0]


def library_directories(
    root: Path,
    executable: Path,
    platform_name: str,
) -> list[Path]:
    directories = {executable.parent.resolve()}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.casefold()
        if platform_name == "windows" and name.endswith(".dll"):
            directories.add(path.parent.resolve())
        elif platform_name == "linux" and ".so" in name:
            directories.add(path.parent.resolve())
        elif platform_name == "macos" and (name.endswith(".dylib") or ".so" in name):
            directories.add(path.parent.resolve())
    return sorted(
        directories,
        key=lambda path: (path != executable.parent.resolve(), str(path)),
    )


def runtime_environment(
    library_dirs: list[Path],
    platform_name: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    prefix = os.pathsep.join(str(path) for path in library_dirs)
    environment["PATH"] = prefix + os.pathsep + environment.get("PATH", "")
    if platform_name == "linux":
        variable = "LD_LIBRARY_PATH"
        environment[variable] = prefix + os.pathsep + environment.get(variable, "")
    elif platform_name == "macos":
        variable = "DYLD_LIBRARY_PATH"
        environment[variable] = prefix + os.pathsep + environment.get(variable, "")
    return environment


def validate_runtime(
    context: InstallerContext,
    executable: Path,
    library_dirs: list[Path],
    package: RuntimePackage,
) -> tuple[str, str]:
    if context.platform_name != "windows":
        executable.chmod(executable.stat().st_mode | 0o111)
    environment = runtime_environment(library_dirs, context.platform_name)
    process_options = {}
    if context.platform_name == "windows":
        process_options["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )

    def run(argument: str) -> str:
        try:
            completed = subprocess.run(
                [str(executable), argument],
                # The extracted runtime directory can inherit restrictive Windows
                # ACLs. The executable directory remains on PATH, so validation does
                # not need to use it as the child working directory.
                cwd=context.root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                **process_options,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(f"Unable to run {executable.name}: {error}") from error
        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{executable.name} {argument} failed with exit code "
                f"{completed.returncode}:\n{output[-3000:]}"
            )
        return output

    version_output = run("--version")
    devices_output = run("--list-devices")
    upper_devices = devices_output.upper()
    if package.device_markers and not any(
        marker in upper_devices for marker in package.device_markers
    ):
        expected = ", ".join(package.device_markers)
        raise RuntimeError(
            f"The {package.description} runtime started but did not expose a compatible "
            f"device. Expected one of: {expected}.\n{devices_output[-3000:]}"
        )
    return version_output, devices_output


def _assert_generated_path(context: InstallerContext, path: Path) -> None:
    try:
        path.resolve().relative_to(context.runtime_root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Refusing to modify a path outside {context.runtime_root}: {path}"
        ) from error


def _remove_generated_tree(context: InstallerContext, path: Path) -> None:
    _assert_generated_path(context, path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def remove_generated_tree(context: InstallerContext, path: Path) -> None:
    _remove_generated_tree(context, path)


def _replace_directory(context: InstallerContext, source: Path, target: Path) -> None:
    _assert_generated_path(context, source)
    _assert_generated_path(context, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f"{target.name}.backup-{uuid.uuid4().hex}")
    had_target = target.exists()
    try:
        if had_target:
            os.replace(target, backup)
        os.replace(source, target)
    except Exception:
        if had_target and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup.exists():
            _remove_generated_tree(context, backup)


def replace_generated_directory(
    context: InstallerContext,
    source: Path,
    target: Path,
) -> None:
    _replace_directory(context, source, target)


def install_package(
    context: InstallerContext,
    package: RuntimePackage,
    cache_dir: Path,
    force: bool,
    offline: bool,
) -> tuple[Path, list[Path], str, str]:
    target = (
        context.runtime_root
        / f"{context.platform_name}-{context.architecture}"
        / package.backend
        / context.release_tag
    )
    if target.exists() and not force:
        try:
            executable = find_server(target, context.executable_name)
            library_dirs = library_directories(
                target, executable, context.platform_name
            )
            version, devices = validate_runtime(
                context, executable, library_dirs, package
            )
            print(f"[OK] Reusing validated runtime: {target}")
            return executable, library_dirs, version, devices
        except RuntimeError as error:
            print(f"[WARN] Existing runtime is invalid and will be replaced: {error}")

    archives = [download_asset(asset, cache_dir, offline) for asset in package.assets]
    staging_parent = context.runtime_root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{package.backend}-", dir=staging_parent))
    payload = staging / "payload"
    payload.mkdir()
    try:
        for archive in archives:
            print(f"[INFO] Extracting {archive.name}")
            extract_archive_safely(archive, payload)
        executable = find_server(payload, context.executable_name)
        library_dirs = library_directories(
            payload, executable, context.platform_name
        )
        print(f"[INFO] Validating {package.description} runtime")
        version, devices = validate_runtime(
            context, executable, library_dirs, package
        )

        executable_relative = executable.relative_to(payload)
        library_relatives = [path.relative_to(payload) for path in library_dirs]
        _replace_directory(context, payload, target)
        executable = target / executable_relative
        library_dirs = [target / path for path in library_relatives]
        print(f"[OK] Runtime installed: {target}")
        return executable, library_dirs, version, devices
    finally:
        if staging.exists():
            _remove_generated_tree(context, staging)


def write_runtime_config(
    context: InstallerContext,
    package: RuntimePackage,
    hardware: HardwareInfo,
    executable: Path,
    library_dirs: list[Path],
    version_output: str,
    devices_output: str,
    source: str = "official ggml-org/llama.cpp GitHub Release",
) -> None:
    detected_cuda = hardware.cuda_selection_version
    if package.cuda_version is None:
        cuda_selection: dict[str, object] = {"match": "not_applicable"}
    else:
        # Recording how the archive was matched keeps CUDA startup reports
        # distinguishable between "the driver really is 13.5" and "the driver is
        # 13.4 and a same-major archive was accepted".
        cuda_selection = {
            "archive_cuda_version": format_cuda_version(package.cuda_version),
            "driver_or_toolkit": format_cuda_version(detected_cuda),
            "match": (
                "undetected"
                if detected_cuda is None
                else cuda_match_mode(package.cuda_version, detected_cuda)
            ),
        }
    config = {
        "schema_version": 1,
        "platform": context.platform_name,
        "architecture": context.architecture,
        "backend": package.backend,
        "backend_description": package.description,
        "llama_cpp_tag": context.release_tag,
        "llama_cpp_commit": context.release_commit,
        "executable": executable.resolve().relative_to(context.root).as_posix(),
        "library_dirs": [
            path.resolve().relative_to(context.root).as_posix()
            for path in library_dirs
        ],
        "runtime_options": {
            "n_gpu_layers": 0 if package.backend == "cpu" else "auto",
            "fit": package.backend != "cpu",
            "fit_target_mib": 1536,
            "flash_attention": "auto",
        },
        "detected_gpus": list(hardware.gpu_names),
        "cuda_selection": cuda_selection,
        "detected_cuda": {
            "driver_compatibility": (
                format_cuda_version(hardware.cuda_driver_version)
                if hardware.cuda_driver_version is not None
                else None
            ),
            "toolkit": (
                format_cuda_version(hardware.cuda_toolkit_version)
                if hardware.cuda_toolkit_version is not None
                else None
            ),
        },
        "version_output": version_output.strip(),
        "devices_output": devices_output.strip(),
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    temporary = context.config_path.with_name(
        f"{context.config_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, context.config_path)
    print(f"[OK] Runtime configuration written: {context.config_path}")


# Platform and GPU detection plus backend routing.
def normalize_platform(system_name: str | None = None) -> str:
    value = (system_name or platform.system()).strip().casefold()
    names = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "macos",
        "macos": "macos",
    }
    if value not in names:
        raise RuntimeError(
            f"Unsupported operating system: {system_name or platform.system()}"
        )
    return names[value]


def normalize_architecture(
    machine: str | None = None,
    platform_name: str | None = None,
) -> str:
    value = (machine or platform.machine()).strip().casefold()
    if value in {"amd64", "x86_64", "x64"}:
        return "x64"
    if value in {"arm64", "aarch64"}:
        return "arm64"
    active_platform = platform_name or normalize_platform()
    raise RuntimeError(
        f"Unsupported {active_platform} architecture: {value or 'unknown'}"
    )


def parse_cuda_driver_version(output: str) -> tuple[int, int] | None:
    match = re.search(
        r"\bCUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)",
        output,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_cuda_toolkit_version(output: str) -> tuple[int, int] | None:
    for pattern in (
        r"\brelease\s+(\d+)\.(\d+)",
        r"\bV(\d+)\.(\d+)(?:\.\d+)?\b",
    ):
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def format_cuda_version(version: tuple[int, int] | None) -> str:
    return "unknown" if version is None else f"{version[0]}.{version[1]}"


def cuda_match_mode(
    required: tuple[int, int], detected: tuple[int, int]
) -> str | None:
    # How a CUDA archive's build version lines up with the detected driver.
    # NVIDIA guarantees minor version compatibility within one major release, and
    # the release archives carry their own cudart, so an archive built for 13.5
    # runs on a driver that reports 13.4. Crossing majors is not covered by that
    # promise, so an older archive still needs a driver at least as new.
    if required[0] == detected[0]:
        return "exact" if required == detected else "same_major"
    if required <= detected:
        return "exact"
    return None


def _find_cuda_compiler_optional() -> Path | None:
    candidates: list[Path] = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc))
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.extend(
                (Path(root) / "bin" / "nvcc", Path(root) / "bin" / "nvcc.exe")
            )
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def detect_cuda_toolkit_version() -> tuple[int, int] | None:
    compiler = _find_cuda_compiler_optional()
    if compiler is None:
        return None
    return parse_cuda_toolkit_version(run_probe([str(compiler), "--version"]))


def detect_windows_hardware(machine: str | None = None) -> HardwareInfo:
    architecture = normalize_architecture(machine, "windows")
    outputs: list[str] = []
    nvidia = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        outputs.append(nvidia)
    nvidia_status = run_probe(["nvidia-smi"])

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell:
        gpu_names = run_probe(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ]
        )
        if gpu_names:
            outputs.append(gpu_names)

    if not outputs:
        wmic = shutil.which("wmic.exe") or shutil.which("wmic")
        if wmic:
            outputs.append(
                run_probe([wmic, "path", "win32_VideoController", "get", "name"])
            )

    names = unique_lines("\n".join(outputs))
    names = [name for name in names if name.casefold() not in {"name", "n/a"}]
    lowered = "\n".join(names).casefold()
    vendors: list[str] = []
    if nvidia or any(token in lowered for token in ("nvidia", "geforce", "quadro")):
        vendors.append("nvidia")
    if any(
        token in lowered
        for token in ("amd ", "amd,", "radeon", "advanced micro devices")
    ):
        vendors.append("amd")
    if "intel" in lowered or " arc " in f" {lowered} ":
        vendors.append("intel")
    if "qualcomm" in lowered or "adreno" in lowered:
        vendors.append("qualcomm")
    return HardwareInfo(
        architecture=architecture,
        gpu_names=tuple(names) or ("No GPU name detected",),
        vendors=tuple(vendors),
        cuda_driver_version=parse_cuda_driver_version(nvidia_status),
        cuda_toolkit_version=detect_cuda_toolkit_version(),
    )


def _linux_display_devices() -> list[str]:
    outputs: list[str] = []
    nvidia = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        outputs.append(
            "\n".join(f"NVIDIA: {line}" for line in nvidia.splitlines())
        )

    if shutil.which("lspci"):
        pci_output = run_probe(["lspci", "-nn"])
        display_lines = [
            line
            for line in pci_output.splitlines()
            if any(
                label in line.casefold()
                for label in ("vga", "3d controller", "display")
            )
        ]
        if display_lines:
            outputs.append("\n".join(display_lines))

    vendor_names = {
        "0x10de": "NVIDIA GPU",
        "0x1002": "AMD GPU",
        "0x8086": "Intel GPU",
    }
    for vendor_path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            vendor = vendor_path.read_text(encoding="ascii").strip().casefold()
        except OSError:
            continue
        if vendor in vendor_names:
            outputs.append(vendor_names[vendor])
    return unique_lines("\n".join(outputs))


def detect_linux_hardware(machine: str | None = None) -> HardwareInfo:
    architecture = normalize_architecture(machine, "linux")
    names = _linux_display_devices()
    lowered = "\n".join(names).casefold()
    vendors: list[str] = []
    if any(token in lowered for token in ("nvidia", "geforce", "quadro")):
        vendors.append("nvidia")
    if any(token in lowered for token in ("amd ", "radeon", "advanced micro devices")):
        vendors.append("amd")
    if "intel" in lowered or " arc " in f" {lowered} ":
        vendors.append("intel")
    return HardwareInfo(
        architecture=architecture,
        gpu_names=tuple(names) or ("No GPU name detected",),
        vendors=tuple(vendors),
        cuda_driver_version=parse_cuda_driver_version(run_probe(["nvidia-smi"])),
        cuda_toolkit_version=detect_cuda_toolkit_version(),
    )


def _macos_gpu_names() -> list[str]:
    system_profiler = shutil.which("system_profiler")
    if not system_profiler:
        return []
    output = run_probe([system_profiler, "SPDisplaysDataType", "-json"], timeout=30)
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    names: list[str] = []
    for display in payload.get("SPDisplaysDataType", []):
        if not isinstance(display, dict):
            continue
        name = display.get("sppci_model") or display.get("_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return unique_lines("\n".join(names))


def detect_macos_hardware(machine: str | None = None) -> HardwareInfo:
    architecture = normalize_architecture(machine, "macos")
    names = _macos_gpu_names()
    return HardwareInfo(
        architecture=architecture,
        gpu_names=tuple(names) or ("No GPU name detected",),
    )


def detect_hardware(platform_name: str) -> HardwareInfo:
    detectors = {
        "windows": detect_windows_hardware,
        "linux": detect_linux_hardware,
        "macos": detect_macos_hardware,
    }
    return detectors[platform_name]()


def compatible_cuda_backends(
    packages: dict[str, RuntimePackage],
    hardware: HardwareInfo,
) -> list[str]:
    versioned = sorted(
        (
            (package.cuda_version, backend)
            for backend, package in packages.items()
            if package.cuda_version is not None
        ),
        reverse=True,
    )
    if not versioned:
        return []
    detected = hardware.cuda_selection_version
    if detected is None:
        # Keep the previous best-effort behavior when nvidia-smi and nvcc do not
        # expose a version. Validation still rejects an unusable runtime.
        return [versioned[-1][1]]
    # A driver is allowed to serve any archive in its own major series, so a
    # driver reporting 13.4 still receives the 13.5 archive. `versioned` is
    # already ordered newest first, so the closest archive is tried before any
    # older-major fallback.
    return [
        backend
        for required, backend in versioned
        if cuda_match_mode(required, detected) is not None
    ]


def select_windows_backend_candidates(
    hardware: HardwareInfo,
    requested_backend: str = "auto",
    allow_fallback: bool = True,
) -> list[str]:
    packages = WINDOWS_PACKAGES[hardware.architecture]
    if requested_backend != "auto":
        if requested_backend == "cuda":
            cuda_backends = compatible_cuda_backends(packages, hardware)
            if not cuda_backends:
                detected = format_cuda_version(hardware.cuda_selection_version)
                raise RuntimeError(
                    f"No compatible Windows CUDA archive is available for detected "
                    f"CUDA {detected} on {hardware.architecture}."
                )
            return [cuda_backends[0]]
        if requested_backend not in packages:
            available = ", ".join(packages)
            raise RuntimeError(
                f"Backend '{requested_backend}' is unavailable for Windows "
                f"{hardware.architecture}. Available: {available}"
            )
        return [requested_backend]

    candidates: list[str] = []

    def add(backend: str) -> None:
        if backend in packages and backend not in candidates:
            candidates.append(backend)

    if hardware.architecture == "x64":
        if "nvidia" in hardware.vendors:
            for backend in compatible_cuda_backends(packages, hardware):
                add(backend)
        if "amd" in hardware.vendors:
            add("rocm")
        if "intel" in hardware.vendors:
            add("sycl")
        add("vulkan")
        add("cpu")
    else:
        if "nvidia" in hardware.vendors:
            for backend in compatible_cuda_backends(packages, hardware):
                add(backend)
        if "qualcomm" in hardware.vendors or not hardware.vendors:
            add("opencl")
        add("cpu")
    return candidates if allow_fallback else candidates[:1]


def select_linux_backend_candidates(
    hardware: HardwareInfo,
    requested_backend: str = "auto",
    allow_fallback: bool = True,
    build_from_source: bool = False,
) -> list[str]:
    packages = LINUX_PACKAGES[hardware.architecture]
    if requested_backend != "auto":
        if requested_backend == "cuda":
            if build_from_source:
                return ["cuda"]
            cuda_backends = compatible_cuda_backends(packages, hardware)
            if not cuda_backends:
                detected = format_cuda_version(hardware.cuda_selection_version)
                raise RuntimeError(
                    f"No compatible Linux CUDA archive is available for detected "
                    f"CUDA {detected} on {hardware.architecture}. Use "
                    f"--backend cuda --build-from-source or another backend."
                )
            return [cuda_backends[0]]
        if requested_backend not in packages:
            available = "cuda (auto version or source build), " + ", ".join(packages)
            raise RuntimeError(
                f"Backend '{requested_backend}' is unavailable for Linux "
                f"{hardware.architecture}. Available: {available}."
            )
        return [requested_backend]

    candidates: list[str] = []

    def add(backend: str) -> None:
        if backend in packages and backend not in candidates:
            candidates.append(backend)

    if "nvidia" in hardware.vendors:
        for backend in compatible_cuda_backends(packages, hardware):
            add(backend)
    if "amd" in hardware.vendors:
        add("rocm")
    if "intel" in hardware.vendors:
        add("sycl")
    add("vulkan")
    add("cpu")
    return candidates if allow_fallback else candidates[:1]


def select_macos_backend_candidates(
    hardware: HardwareInfo,
    requested_backend: str = "auto",
    allow_fallback: bool = True,
) -> list[str]:
    packages = MACOS_PACKAGES[hardware.architecture]
    if requested_backend != "auto":
        if requested_backend not in packages:
            available = ", ".join(packages)
            raise RuntimeError(
                f"Backend '{requested_backend}' is unavailable for macOS "
                f"{hardware.architecture}. Available: {available}"
            )
        return [requested_backend]
    candidates = ["metal", "cpu"]
    return candidates if allow_fallback else candidates[:1]


def select_backend_candidates(
    platform_name: str,
    hardware: HardwareInfo,
    requested_backend: str = "auto",
    allow_fallback: bool = True,
    build_from_source: bool = False,
) -> list[str]:
    if platform_name == "windows":
        return select_windows_backend_candidates(
            hardware, requested_backend, allow_fallback
        )
    if platform_name == "linux":
        return select_linux_backend_candidates(
            hardware,
            requested_backend,
            allow_fallback,
            build_from_source,
        )
    return select_macos_backend_candidates(
        hardware, requested_backend, allow_fallback
    )


# Linux CUDA source build support.
def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _find_executable(candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError(f"Missing {label}. Install it and ensure it is available in PATH.")


def _find_cuda_compiler() -> Path:
    compiler = _find_cuda_compiler_optional()
    if compiler is not None:
        return compiler
    raise RuntimeError(
        "CUDA compiler 'nvcc' was not found. Install the NVIDIA CUDA Toolkit and "
        "add its bin directory to PATH or set CUDA_HOME."
    )


def _cuda_library_directories(
    cuda_root: Path,
    architecture: str,
) -> tuple[Path, ...]:
    target_arch = "x86_64-linux" if architecture == "x64" else "aarch64-linux"
    candidates = (
        cuda_root / "lib64",
        cuda_root / "targets" / target_arch / "lib",
    )
    return tuple(path.resolve() for path in candidates if path.is_dir())


def detect_cuda_build_tools(architecture: str) -> CudaBuildTools:
    cuda_compiler = _find_cuda_compiler()
    cuda_root = cuda_compiler.parent.parent.resolve()
    tools = CudaBuildTools(
        git=_find_executable(("git",), "Git"),
        cmake=_find_executable(("cmake",), "CMake"),
        cuda_compiler=cuda_compiler,
        cuda_root=cuda_root,
        cuda_library_dirs=_cuda_library_directories(cuda_root, architecture),
        c_compiler=_find_executable(("cc", "gcc", "clang"), "a C compiler"),
        cxx_compiler=_find_executable(("c++", "g++", "clang++"), "a C++ compiler"),
    )
    git_version = run_probe([tools.git, "--version"])
    cmake_version = run_probe([tools.cmake, "--version"])
    cuda_version = run_probe([str(tools.cuda_compiler), "--version"])
    if not git_version or not cmake_version or not cuda_version:
        raise RuntimeError(
            "Git, CMake, or nvcc was found but could not be executed successfully."
        )
    print(f"[INFO] Git: {git_version.splitlines()[0]}")
    print(f"[INFO] CMake: {cmake_version.splitlines()[0]}")
    print(f"[INFO] CUDA compiler: {cuda_version.splitlines()[-1]}")
    print(f"[INFO] C compiler: {tools.c_compiler}")
    print(f"[INFO] C++ compiler: {tools.cxx_compiler}")
    return tools


def _run_checked(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> str:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=capture_output,
            text=True,
            encoding="utf-8" if capture_output else None,
            errors="replace" if capture_output else None,
        )
    except OSError as error:
        raise RuntimeError(f"Unable to run {arguments[0]}: {error}") from error
    output = ""
    if capture_output:
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
    if completed.returncode != 0:
        detail = f"\n{output[-4000:]}" if output else ""
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(arguments)}{detail}"
        )
    return output


def _verified_source_checkout(source_dir: Path, git: str) -> bool:
    if not (source_dir / ".git").is_dir():
        return False
    head_output = _run_checked(
        [git, "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True,
    )
    if not head_output:
        return False
    if head_output.splitlines()[0] != RELEASE_COMMIT_FULL:
        return False
    status = _run_checked(
        [git, "-C", str(source_dir), "status", "--porcelain"],
        capture_output=True,
    )
    return not status


def prepare_source_checkout(
    context: InstallerContext,
    git: str,
    source_dir: Path,
    offline: bool,
) -> Path:
    source_dir = source_dir.resolve()
    source_root = (context.runtime_root / ".sources").resolve()
    try:
        relative_source = source_dir.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            f"Source cache must be inside {source_root}: {source_dir}"
        ) from error
    if not relative_source.parts:
        raise RuntimeError(f"Source cache cannot be the source root itself: {source_dir}")

    if source_dir.exists():
        try:
            if _verified_source_checkout(source_dir, git):
                print(f"[OK] Reusing verified source checkout: {source_dir}")
                return source_dir
        except RuntimeError as error:
            print(f"[WARN] Unable to verify cached source checkout: {error}")
        if offline:
            raise RuntimeError(
                f"Offline source cache is not a clean {RELEASE_COMMIT_FULL} checkout: "
                f"{source_dir}"
            )
        print("[WARN] Replacing an invalid cached source checkout")
        remove_generated_tree(context, source_dir)
    elif offline:
        raise RuntimeError(f"Offline source cache does not exist: {source_dir}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = source_dir.with_name(f"{source_dir.name}.clone-{uuid.uuid4().hex}")
    print(f"[INFO] Cloning llama.cpp {RELEASE_TAG} source")
    started = time.monotonic()
    try:
        _run_checked(
            [
                git,
                "clone",
                "--depth",
                "1",
                "--branch",
                RELEASE_TAG,
                "--single-branch",
                SOURCE_REPOSITORY,
                str(temporary),
            ]
        )
        if not _verified_source_checkout(temporary, git):
            raise RuntimeError(
                f"Downloaded source did not match pinned commit {RELEASE_COMMIT_FULL}"
            )
        os.replace(temporary, source_dir)
    finally:
        if temporary.exists():
            remove_generated_tree(context, temporary)
    print(f"[OK] Source checkout verified in {time.monotonic() - started:.1f} s")
    return source_dir


def cuda_cmake_arguments(
    tools: CudaBuildTools,
    source_dir: Path,
    build_dir: Path,
) -> list[str]:
    runtime_paths = ["$ORIGIN", *(str(path) for path in tools.cuda_library_dirs)]
    rpath = ";".join(runtime_paths)
    return [
        tools.cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DGGML_CUDA=ON",
        "-DGGML_NATIVE=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_COMPILER={tools.cuda_compiler}",
        f"-DCMAKE_C_COMPILER={tools.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={tools.cxx_compiler}",
        f"-DCMAKE_BUILD_RPATH={rpath}",
        f"-DCMAKE_INSTALL_RPATH={rpath}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_UI=OFF",
    ]


def install_cuda_from_source(
    context: InstallerContext,
    package: RuntimePackage,
    source_dir: Path,
    tools: CudaBuildTools,
    jobs: int,
    force: bool,
    offline: bool,
) -> tuple[Path, list[Path], str, str]:
    target = (
        context.runtime_root
        / f"{context.platform_name}-{context.architecture}"
        / "cuda"
        / context.release_tag
    )
    if target.exists() and not force:
        try:
            executable = find_server(target, context.executable_name)
            library_dirs = library_directories(
                target, executable, context.platform_name
            )
            version, devices = validate_runtime(
                context, executable, library_dirs, package
            )
            print(f"[OK] Reusing validated runtime: {target}")
            return executable, library_dirs, version, devices
        except (OSError, RuntimeError) as error:
            print(f"[WARN] Existing CUDA runtime is invalid and will be replaced: {error}")

    source_dir = prepare_source_checkout(context, tools.git, source_dir, offline)
    build_parent = context.runtime_root / ".build"
    staging_parent = context.runtime_root / ".staging"
    build_parent.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix="cuda-", dir=build_parent))
    staging = Path(tempfile.mkdtemp(prefix="cuda-source-", dir=staging_parent))
    payload = staging / "payload"
    started = time.monotonic()
    try:
        print("[INFO] Configuring llama.cpp CUDA build")
        _run_checked(cuda_cmake_arguments(tools, source_dir, build_dir))
        print(f"[INFO] Building llama-server with {jobs} parallel jobs")
        _run_checked(
            [
                tools.cmake,
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "llama-server",
                "-j",
                str(jobs),
            ]
        )
        built_bin = build_dir / "bin"
        if not (built_bin / context.executable_name).is_file():
            raise RuntimeError(
                f"CUDA build completed without {built_bin / context.executable_name}"
            )
        shutil.copytree(built_bin, payload, symlinks=True)
        executable = payload / context.executable_name
        library_dirs = library_directories(payload, executable, context.platform_name)
        print("[INFO] Validating locally built CUDA runtime")
        version, devices = validate_runtime(
            context, executable, library_dirs, package
        )
        executable_relative = executable.relative_to(payload)
        library_relatives = [path.relative_to(payload) for path in library_dirs]
        replace_generated_directory(context, payload, target)
        executable = target / executable_relative
        library_dirs = [target / path for path in library_relatives]
        print(
            f"[OK] CUDA runtime built and installed in "
            f"{time.monotonic() - started:.1f} s: {target}"
        )
        return executable, library_dirs, version, devices
    finally:
        if staging.exists():
            remove_generated_tree(context, staging)
        if build_dir.exists():
            remove_generated_tree(context, build_dir)


# Cross-platform command-line entry point.
def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the operating system and GPU, then install a verified local "
            "llama-server runtime for ComfyUI Qwen H3 Prompt."
        )
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help=(
            "Backend to install. Use auto for platform detection; run "
            "--list-backends to see choices for this machine."
        ),
    )
    parser.add_argument(
        "--build-from-source",
        action="store_true",
        help="Build Linux CUDA locally; requires --backend cuda.",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_integer,
        default=min(os.cpu_count() or 1, 8),
        help="Parallel jobs for a Linux CUDA source build (default: up to 8).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "runtime" / ".sources" / f"llama.cpp-{RELEASE_TAG}",
        help=(
            "Cached llama.cpp checkout for a Linux CUDA source build; it must be "
            "inside runtime/.sources."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if a valid runtime already exists.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use verified release archives or a cached source checkout without "
            "network access."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "runtime" / ".downloads" / RELEASE_TAG,
        help="Directory used to cache downloaded release archives.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="In auto mode, do not try another backend when validation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the detected platform and selected runtime without installing it.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List backends available for the detected platform and architecture.",
    )
    return parser.parse_args(argv)


def _platform_label(platform_name: str) -> str:
    return {"windows": "Windows", "linux": "Linux", "macos": "macOS"}[
        platform_name
    ]


def _print_backends(platform_name: str, architecture: str) -> None:
    label = _platform_label(platform_name)
    print(f"Available {label} {architecture} backends:")
    if platform_name == "linux":
        cuda_description = LINUX_CUDA_PACKAGES[architecture].description
        print(f"  {'cuda':12} Auto-select a compatible CUDA archive")
        print(
            f"  {'cuda':12} {cuda_description} "
            f"[with --build-from-source]"
        )
    for backend, package in PLATFORM_PACKAGES[platform_name][architecture].items():
        print(f"  {backend:12} {package.description}")


def _print_detection(platform_name: str, hardware: HardwareInfo) -> None:
    label = _platform_label(platform_name)
    print(f"[INFO] Operating system: {label}")
    print(f"[INFO] Architecture: {hardware.architecture}")
    if platform_name != "macos":
        print(f"[INFO] Detected GPU vendors: {', '.join(hardware.vendors) or 'unknown'}")
    for name in hardware.gpu_names:
        print(f"[INFO] GPU: {name}")
    if "nvidia" in hardware.vendors:
        print(
            "[INFO] NVIDIA driver CUDA compatibility: "
            f"{format_cuda_version(hardware.cuda_driver_version)}"
        )
        print(
            "[INFO] CUDA Toolkit: "
            f"{format_cuda_version(hardware.cuda_toolkit_version)}"
        )


def _print_dry_run(
    platform_name: str,
    hardware: HardwareInfo,
    candidates: list[str],
    source_dir: Path,
) -> None:
    packages = PLATFORM_PACKAGES[platform_name][hardware.architecture]
    for backend in candidates:
        if platform_name == "linux" and backend == "cuda":
            package = LINUX_CUDA_PACKAGES[hardware.architecture]
            print(f"[DRY RUN] cuda: {package.description}")
            print(f"  repository={SOURCE_REPOSITORY}")
            print(f"  tag={RELEASE_TAG}")
            print(f"  commit={RELEASE_COMMIT_FULL}")
            print(f"  source_cache={source_dir.resolve()}")
            print("  cmake=-DGGML_CUDA=ON -DGGML_NATIVE=OFF")
            continue
        package = packages[backend]
        print(f"[DRY RUN] {backend}: {package.description}")
        for asset in package.assets:
            print(f"  {asset.url}")
            print(f"  sha256={asset.sha256}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        platform_name = normalize_platform()
        hardware = detect_hardware(platform_name)
    except Exception as error:
        print_error(f"[ERROR] Platform or hardware detection failed: {error}")
        return 1

    if arguments.list_backends:
        _print_backends(platform_name, hardware.architecture)
        return 0

    if arguments.build_from_source and not (
        platform_name == "linux" and arguments.backend.casefold() == "cuda"
    ):
        print_error(
            "[ERROR] --build-from-source requires Linux with --backend cuda."
        )
        return 1

    _print_detection(platform_name, hardware)
    try:
        candidates = select_backend_candidates(
            platform_name,
            hardware,
            requested_backend=arguments.backend.casefold(),
            allow_fallback=not arguments.no_fallback,
            build_from_source=arguments.build_from_source,
        )
    except RuntimeError as error:
        print_error(f"[ERROR] {error}")
        return 1
    print(f"[INFO] Backend order: {' -> '.join(candidates)}")

    if arguments.dry_run:
        _print_dry_run(
            platform_name,
            hardware,
            candidates,
            arguments.source_dir,
        )
        return 0

    context = InstallerContext(
        root=ROOT,
        platform_name=platform_name,
        architecture=hardware.architecture,
        release_tag=RELEASE_TAG,
        release_commit=RELEASE_COMMIT,
        executable_name=(
            "llama-server.exe" if platform_name == "windows" else "llama-server"
        ),
    )
    packages = PLATFORM_PACKAGES[platform_name][hardware.architecture]
    errors: list[str] = []
    for index, backend in enumerate(candidates):
        package = (
            LINUX_CUDA_PACKAGES[hardware.architecture]
            if platform_name == "linux" and backend == "cuda"
            else packages[backend]
        )
        if index:
            print(f"[WARN] Trying fallback backend: {backend}")
        try:
            if platform_name == "linux" and backend == "cuda":
                if "nvidia" not in hardware.vendors:
                    raise RuntimeError(
                        "No NVIDIA GPU was detected. CUDA source builds require an "
                        "NVIDIA driver and a visible NVIDIA GPU."
                    )
                tools = detect_cuda_build_tools(hardware.architecture)
                executable, library_dirs, version, devices = install_cuda_from_source(
                    context,
                    package,
                    arguments.source_dir.resolve(),
                    tools,
                    arguments.jobs,
                    arguments.force,
                    arguments.offline,
                )
            else:
                executable, library_dirs, version, devices = install_package(
                    context,
                    package,
                    arguments.cache_dir.resolve(),
                    arguments.force,
                    arguments.offline,
                )
            write_runtime_config(
                context,
                package,
                hardware,
                executable,
                library_dirs,
                version,
                devices,
                source=(
                    SOURCE_CONFIG_VALUE
                    if platform_name == "linux" and backend == "cuda"
                    else "official ggml-org/llama.cpp GitHub Release"
                ),
            )
            label = _platform_label(platform_name)
            print_success(
                f"[OK] {label} llama.cpp runtime setup complete: "
                f"{package.description}"
            )
            if package.backend == "cpu":
                print("[WARN] CPU mode is functional but Qwen3.8-27B will be slow.")
            return 0
        except Exception as error:
            message = f"{backend}: {error}"
            errors.append(message)
            print_error(f"[ERROR] {message}")

    label = _platform_label(platform_name)
    details = "\n".join(f"  - {message}" for message in errors)
    print_error(f"[ERROR] No {label} runtime backend could be installed:\n{details}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
