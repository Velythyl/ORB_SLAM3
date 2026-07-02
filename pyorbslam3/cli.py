from __future__ import annotations

import argparse
import subprocess

from . import __version__
from .docker import ContainerOptions, docker_executable, resolve_image, run_container


def _run_container(args: argparse.Namespace, command: list[str]) -> int:
    try:
        completed = run_container(
            command,
            ContainerOptions(
                image=args.image,
                workdir=args.workdir,
                mounts=tuple(args.mount),
                devices=tuple(args.device),
                env=tuple(args.env),
                gpu=args.gpu,
                name=args.name,
                user=args.user,
                privileged=args.privileged,
                no_display=args.no_display,
            ),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return completed.returncode


def cmd_image(args: argparse.Namespace) -> int:
    print(resolve_image(args.image))
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    try:
        docker = docker_executable()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return subprocess.call([docker, "pull", resolve_image(args.image)])


def cmd_run(args: argparse.Namespace) -> int:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    return _run_container(args, command)


def _add_container_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", help="Container image to run. Defaults to ORB_SLAM3_IMAGE or the GHCR image.")
    parser.add_argument("--workdir", default="/work", help="Working directory inside the container.")
    parser.add_argument("--mount", action="append", default=[], help="Additional Docker volume mount, for example /data:/data:ro.")
    parser.add_argument("--device", action="append", default=[], help="Device to pass through to Docker, for example /dev/video0.")
    parser.add_argument("--env", "-e", action="append", default=[], help="Environment variable to pass through to Docker.")
    parser.add_argument("--gpu", help="Value for Docker --gpus, for example all.")
    parser.add_argument("--name", help="Optional Docker container name.")
    parser.add_argument("--user", help="User to run as inside the container, for example 1000:1000.")
    parser.add_argument("--privileged", action="store_true", help="Run the container in privileged mode.")
    parser.add_argument("--no-display", action="store_true", help="Do not forward local display environment variables.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orb-slam3", description="Run ORB-SLAM3 from the published Docker image.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command_name")

    image = subparsers.add_parser("image", help="Print the container image used by default.")
    image.add_argument("--image")
    image.set_defaults(func=cmd_image)

    pull = subparsers.add_parser("pull", help="Pull the ORB-SLAM3 container image.")
    pull.add_argument("--image")
    pull.set_defaults(func=cmd_pull)

    run = subparsers.add_parser("run", help="Run a command in the ORB-SLAM3 container.")
    _add_container_options(run)
    run.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, for example mono_tum ...")
    run.set_defaults(func=cmd_run)

    shell = subparsers.add_parser("shell", help="Open an interactive shell in the ORB-SLAM3 container.")
    _add_container_options(shell)
    shell.set_defaults(func=lambda args: _run_container(args, ["bash"]))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
