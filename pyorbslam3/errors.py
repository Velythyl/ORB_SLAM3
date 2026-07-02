from __future__ import annotations

from pathlib import Path
from typing import Sequence


class OrbSlam3RunError(RuntimeError):
    """Raised when an ORB-SLAM3 container command fails."""

    returncode: int
    command: tuple[str, ...]
    log: Path | None

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        command: Sequence[str],
        log: Path | None = None,
    ) -> None:
        if log is not None:
            message = f"{message} See log: {log}"
        super().__init__(message)
        self.returncode = returncode
        self.command = tuple(command)
        self.log = log
