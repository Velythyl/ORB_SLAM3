from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pyorbslam3.docker import ContainerOptions, build_docker_command


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
