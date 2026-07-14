from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pyorbslam3.docker import ContainerOptions, build_docker_command
from pyorbslam3.docker import docker_executable


@patch("pyorbslam3.docker.docker_executable", return_value="/usr/bin/docker")
def test_builds_basic_command(_docker: object) -> None:
    command = build_docker_command(
        ["help"],
        ContainerOptions(
            image="orb-slam3:local",
            workdir="/work/run",
            mount_cwd=Path("/repo"),
            no_display=True,
            interactive=False,
        ),
    )

    assert command == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "-v",
        "/repo:/work",
        "-w",
        "/work/run",
        "orb-slam3:local",
        "help",
    ]


@patch("pyorbslam3.docker.docker_executable", return_value="/usr/bin/docker")
def test_includes_extra_runtime_options(_docker: object) -> None:
    command = build_docker_command(
        ["bash"],
        ContainerOptions(
            image="orb-slam3:local",
            mount_cwd=None,
            mounts=("/data:/data:ro",),
            devices=("/dev/video0",),
            env=("A=B",),
            env_vars={"C": "D"},
            gpu="all",
            user="1000:1000",
            privileged=True,
            no_display=True,
            interactive=False,
        ),
    )

    assert "--gpus" in command
    assert "/data:/data:ro" in command
    assert "/dev/video0" in command
    assert "A=B" in command
    assert "C=D" in command
    assert "1000:1000" in command
    assert "--privileged" in command


@patch("pyorbslam3.docker.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "podman" else None)
def test_falls_back_to_podman_when_docker_is_missing(_which: object) -> None:
    assert docker_executable() == "/usr/bin/podman"


@patch("pyorbslam3.docker.shutil.which", return_value="/custom/runtime")
def test_container_runtime_env_override(_which: object, monkeypatch) -> None:
    monkeypatch.setenv("ORB_SLAM3_CONTAINER_RUNTIME", "podman")

    assert docker_executable() == "/custom/runtime"
