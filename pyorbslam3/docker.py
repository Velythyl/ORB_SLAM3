from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_IMAGE = "ghcr.io/velythyl/orb_slam3:latest"


@dataclass(frozen=True)
class ContainerOptions:
    image: str | None = None
    workdir: str = "/work"
    mount_cwd: Path | None = field(default_factory=Path.cwd)
    mounts: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    env: tuple[str, ...] = ()
    env_vars: Mapping[str, str] = field(default_factory=dict)
    gpu: str | None = None
    name: str | None = None
    user: str | None = None
    privileged: bool = False
    no_display: bool = False
    interactive: bool | None = None


@dataclass(frozen=True)
class CompletedRun:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def docker_executable() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required to run ORB-SLAM3. Install Docker and try again.")
    return docker


def resolve_image(image: str | None = None) -> str:
    return image or os.environ.get("ORB_SLAM3_IMAGE", DEFAULT_IMAGE)


def display_args(no_display: bool) -> list[str]:
    if no_display:
        return []

    args: list[str] = []
    display = os.environ.get("DISPLAY")
    if display:
        args.extend(["-e", f"DISPLAY={display}"])
        if Path("/tmp/.X11-unix").exists():
            args.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])

    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if wayland_display and xdg_runtime_dir and Path(xdg_runtime_dir).exists():
        args.extend(["-e", f"WAYLAND_DISPLAY={wayland_display}", "-e", f"XDG_RUNTIME_DIR={xdg_runtime_dir}"])
        args.extend(["-v", f"{xdg_runtime_dir}:{xdg_runtime_dir}:rw"])

    return args


def build_docker_command(command: Sequence[str], options: ContainerOptions) -> list[str]:
    docker_args = [docker_executable(), "run", "--rm"]

    interactive = sys.stdin.isatty() if options.interactive is None else options.interactive
    if interactive:
        docker_args.append("-it")

    if options.mount_cwd is not None:
        docker_args.extend(["-v", f"{options.mount_cwd.resolve()}:/work"])
    docker_args.extend(["-w", options.workdir])
    docker_args.extend(display_args(options.no_display))

    if options.gpu:
        docker_args.extend(["--gpus", options.gpu])
    if options.privileged:
        docker_args.append("--privileged")
    for device in options.devices:
        docker_args.extend(["--device", device])
    for mount in options.mounts:
        docker_args.extend(["-v", mount])
    for env in options.env:
        docker_args.extend(["-e", env])
    for key, value in options.env_vars.items():
        docker_args.extend(["-e", f"{key}={value}"])
    if options.user:
        docker_args.extend(["--user", options.user])
    if options.name:
        docker_args.extend(["--name", options.name])

    docker_args.append(resolve_image(options.image))
    docker_args.extend(command or ["bash"])
    return docker_args


def run_container(
    command: Sequence[str],
    options: ContainerOptions | None = None,
    *,
    capture_output: bool = False,
) -> CompletedRun:
    options = options or ContainerOptions()
    docker_args = build_docker_command(command, options)
    if capture_output:
        completed = subprocess.run(docker_args, text=True, capture_output=True, check=False)
        return CompletedRun(tuple(docker_args), completed.returncode, completed.stdout, completed.stderr)

    returncode = subprocess.call(docker_args)
    return CompletedRun(tuple(docker_args), returncode)
