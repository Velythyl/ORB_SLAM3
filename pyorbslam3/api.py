from __future__ import annotations

import shlex
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .artifacts import (
    MapPoint,
    Pose,
    PosedObservation,
    read_map_points_csv,
    read_observations_jsonl,
    read_ply_pointcloud,
    read_ply_vertex_count,
    read_tum_trajectory,
    view_pointcloud,
)
from .docker import ContainerOptions, run_container
from .errors import OrbSlam3RunError

DEFAULT_LD_LIBRARY_PATH = (
    "/opt/orbslam3/lib:"
    "/opt/orbslam3/Thirdparty/DBoW2/lib:"
    "/opt/orbslam3/Thirdparty/g2o/lib:"
    "/usr/local/lib"
)

@dataclass(frozen=True)
class ImageFrame:
    timestamp: float
    image: str | Path
    depth: str | Path | None = None


@dataclass(frozen=True)
class StereoFrame:
    timestamp: float
    left: str | Path
    right: str | Path


@dataclass(frozen=True)
class RGBDFrame:
    timestamp: float
    image: str | Path
    depth: str | Path


SequenceFrame = ImageFrame | StereoFrame | RGBDFrame


@dataclass(frozen=True)
class IMUPoint:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class RGBDCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 5000.0


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    keyframe_trajectory: Path | None
    camera_trajectory: Path | None
    atlas: Path | None
    observations_path: Path | None
    map_points_path: Path | None
    log: Path
    poses: tuple[Pose, ...]
    observations: tuple[PosedObservation, ...] = ()
    map_points: tuple[MapPoint, ...] = ()
    runner: "BaseRunner | None" = None
    dataset: Path | None = None
    sensor: str | None = None
    sequence_manifest: Path | None = None

    def make_pointcloud(
        self,
        output: Path,
        *,
        dataset: Path | None = None,
        frames: Iterable[SequenceFrame] | None = None,
        manifest: Path | None = None,
        camera: RGBDCamera | None = None,
        stride: int = 4,
        min_depth: float = 0.25,
        max_depth: float = 4.5,
    ) -> Path:
        if self.runner is None:
            raise RuntimeError("This RunResult is not associated with a runner.")
        if self.keyframe_trajectory is None:
            raise RuntimeError("A keyframe trajectory is required to generate a point cloud.")
        dataset = dataset or self.dataset
        if dataset is None:
            raise RuntimeError("Pass dataset=... because this RunResult does not record one.")
        if self.sensor == "monocular":
            warnings.warn(
                "Generating an RGB-D point cloud from monocular poses can have ambiguous scale.",
                RuntimeWarning,
                stacklevel=2,
            )
        manifest = manifest or self.sequence_manifest
        return self.runner.make_pointcloud(
            dataset=dataset,
            trajectory=self.keyframe_trajectory,
            output=output,
            frames=frames,
            manifest=manifest,
            camera=camera,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    def read_pointcloud(self, path: Path):
        return read_ply_pointcloud(path)

    def view_pointcloud(self, path: Path, *, window_name: str = "ORB-SLAM3 Point Cloud") -> None:
        view_pointcloud(path, window_name=window_name)


class BaseRunner:
    sensor = "unknown"
    manifest_columns = ("timestamp", "image")

    def __init__(
        self,
        *,
        image: str | None = None,
        vocabulary: str | Path = "/opt/orbslam3/Vocabulary/ORBvoc.txt",
        settings: str | Path,
        repo_root: Path | None = None,
    ) -> None:
        self.image = image
        self.vocabulary = vocabulary
        self.settings = settings
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def run_dataset(
        self,
        *,
        dataset: Path,
        output_dir: Path,
        frames: Iterable[SequenceFrame] | None = None,
        manifest: Path | None = None,
        no_display: bool = True,
    ) -> RunResult:
        """Run an explicit timestamped image sequence.

        The dataset directory is only used as the root for image paths listed in
        ``frames`` or ``manifest``.
        """
        return self._run_manifest_sequence(
            dataset=dataset,
            output_dir=output_dir,
            frames=frames,
            manifest=manifest,
            no_display=no_display,
        )

    def run_sequence(
        self,
        *,
        dataset: Path,
        output_dir: Path,
        frames: Iterable[SequenceFrame] | None = None,
        manifest: Path | None = None,
        no_display: bool = True,
    ) -> RunResult:
        """Run an arbitrary timestamped image sequence.

        The sequence manifest format is whitespace-separated:
        ``timestamp image_path [depth_path]``. Paths are relative to ``dataset``.
        RGB-D runs require a depth path for every frame.
        """
        return self.run_dataset(
            dataset=dataset,
            output_dir=output_dir,
            frames=frames,
            manifest=manifest,
            no_display=no_display,
        )

    def build_map(
        self,
        *,
        dataset: Path,
        atlas: Path,
        output_dir: Path,
        no_display: bool = True,
        frames: Iterable[SequenceFrame] | None = None,
        manifest: Path | None = None,
    ) -> RunResult:
        atlas_value, final_atlas = self._atlas_paths(atlas)
        return self._run_manifest_sequence(
            dataset=dataset,
            output_dir=output_dir,
            frames=frames,
            manifest=manifest,
            no_display=no_display,
            settings_append=(f'System.SaveAtlasToFile: "{atlas_value}"',),
            atlas=final_atlas,
        )

    def build_map_sequence(
        self,
        *,
        dataset: Path,
        atlas: Path,
        output_dir: Path,
        frames: Iterable[ImageFrame] | None = None,
        manifest: Path | None = None,
        no_display: bool = True,
    ) -> RunResult:
        return self.build_map(
            dataset=dataset,
            output_dir=output_dir,
            atlas=atlas,
            frames=frames,
            manifest=manifest,
            no_display=no_display,
        )

    def localize(self, *args: object, **kwargs: object) -> RunResult:
        raise NotImplementedError(
            "Localization mode needs the narrow C++ helper described in plans/thin_python_wrapper.md. "
            "Map building and batch trajectory runs are available now."
        )

    def make_pointcloud(
        self,
        *,
        dataset: Path,
        trajectory: Path,
        output: Path,
        frames: Iterable[ImageFrame] | None = None,
        manifest: Path | None = None,
        camera: RGBDCamera | None = None,
        stride: int = 4,
        min_depth: float = 0.25,
        max_depth: float = 4.5,
    ) -> Path | None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        helper = self._ensure_pointcloud_helper(output.parent)
        log = output.with_suffix(output.suffix + ".log")
        pointcloud_manifest = self._sequence_manifest(
            output_dir=output.parent,
            frames=frames,
            manifest=manifest,
            require_input=False,
        )
        command = [
            self._container_path(helper),
            self._container_path(dataset),
            self._container_path(trajectory),
            self._container_path(output),
        ]
        if pointcloud_manifest is not None:
            command.extend(["--manifest", self._container_path(pointcloud_manifest)])
        if camera is not None:
            command.extend(
                [
                    "--fx",
                    str(camera.fx),
                    "--fy",
                    str(camera.fy),
                    "--cx",
                    str(camera.cx),
                    "--cy",
                    str(camera.cy),
                    "--depth-scale",
                    str(camera.depth_scale),
                ]
            )
        command.extend(["--stride", str(stride), "--min-depth", str(min_depth), "--max-depth", str(max_depth)])
        completed = self._container(command, workdir="/work", no_display=True)
        self._write_log(log, completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise OrbSlam3RunError(
                "ORB-SLAM3 point cloud generation failed.",
                returncode=completed.returncode,
                command=completed.args,
                log=log,
            )
        vertex_count = read_ply_vertex_count(output)
        if vertex_count <= 0:
            raise ValueError(f"{output}: generated PLY contains no vertices")
        return output

    def _run_manifest_sequence(
        self,
        *,
        dataset: Path,
        output_dir: Path,
        frames: Iterable[ImageFrame] | None,
        manifest: Path | None,
        no_display: bool,
        settings_append: Sequence[str] = (),
        atlas: Path | None = None,
    ) -> RunResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log = output_dir / "orb_slam3.log"
        settings = self._settings_for_run(output_dir, settings_append)
        sequence_manifest = self._sequence_manifest(output_dir=output_dir, frames=frames, manifest=manifest)
        executable = self._ensure_observation_helper(output_dir)
        command = self._observation_command(
            executable=executable,
            dataset=dataset,
            settings=settings,
            output_dir=output_dir,
            manifest=sequence_manifest,
        )
        return self._execute_run(
            command=command,
            dataset=dataset,
            output_dir=output_dir,
            log=log,
            no_display=no_display,
            atlas=atlas,
            sequence_manifest=sequence_manifest,
        )

    def _observation_command(
        self,
        *,
        executable: str | Path,
        dataset: Path,
        settings: str | Path,
        output_dir: Path,
        manifest: Path | None = None,
    ) -> list[str]:
        command = [
            self._container_path(executable),
            self.sensor,
            self._container_path(self.vocabulary),
            self._container_path(settings),
            self._container_path(dataset),
            self._container_path(output_dir / "observations.jsonl"),
            self._container_path(output_dir / "map_points.csv"),
        ]
        if manifest is not None:
            command.extend(["--manifest", self._container_path(manifest)])
        return command

    def _execute_run(
        self,
        *,
        command: Sequence[str],
        dataset: Path,
        output_dir: Path,
        log: Path,
        no_display: bool,
        atlas: Path | None,
        sequence_manifest: Path | None,
    ) -> RunResult:
        completed = self._container(command, workdir=self._container_path(output_dir), no_display=no_display)
        self._write_log(log, completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise OrbSlam3RunError(
                "ORB-SLAM3 run failed.",
                returncode=completed.returncode,
                command=completed.args,
                log=log,
            )

        keyframe = output_dir / "KeyFrameTrajectory.txt"
        camera = output_dir / "CameraTrajectory.txt"
        keyframe_path = keyframe if keyframe.exists() else None
        camera_path = camera if camera.exists() else None
        trajectory_for_poses = keyframe_path or camera_path
        poses = read_tum_trajectory(trajectory_for_poses) if trajectory_for_poses else ()
        observations_path = output_dir / "observations.jsonl"
        map_points_path = output_dir / "map_points.csv"
        observations = read_observations_jsonl(observations_path) if observations_path.exists() else ()
        map_points = read_map_points_csv(map_points_path) if map_points_path.exists() else ()
        return RunResult(
            output_dir=output_dir,
            keyframe_trajectory=keyframe_path,
            camera_trajectory=camera_path,
            atlas=atlas,
            observations_path=observations_path if observations_path.exists() else None,
            map_points_path=map_points_path if map_points_path.exists() else None,
            log=log,
            poses=poses,
            observations=observations,
            map_points=map_points,
            runner=self,
            dataset=Path(dataset),
            sensor=self.sensor,
            sequence_manifest=sequence_manifest,
        )

    def _sequence_manifest(
        self,
        *,
        output_dir: Path,
        frames: Iterable[SequenceFrame] | None,
        manifest: Path | None,
        require_input: bool = True,
    ) -> Path | None:
        if frames is None and manifest is None and not require_input:
            return None
        if frames is None and manifest is None:
            raise ValueError("Pass either frames=... or manifest=... for arbitrary sequence runs.")
        if frames is not None and manifest is not None:
            raise ValueError("Pass frames=... or manifest=..., not both.")
        if manifest is not None:
            return Path(manifest)

        generated = output_dir / "configs" / "sequence_manifest.txt"
        generated.parent.mkdir(parents=True, exist_ok=True)
        with generated.open("w", encoding="utf-8") as handle:
            for index, frame in enumerate(frames or ()):
                handle.write(self._manifest_row(frame, index))
        return generated

    def _manifest_row(self, frame: SequenceFrame, index: int) -> str:
        if not isinstance(frame, ImageFrame):
            raise ValueError(f"{self.sensor} frame {index} must be an ImageFrame.")
        if frame.depth is not None:
            raise ValueError(f"{self.sensor} frame {index} includes a depth path.")
        return f"{float(frame.timestamp):.9f} {frame.image}\n"

    def _settings_for_run(self, output_dir: Path, settings_append: Sequence[str]) -> str | Path:
        if not settings_append:
            return self.settings
        configs = output_dir / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        generated = configs / "settings.yaml"
        script = (
            f"cp {shlex.quote(self._container_path(self.settings))} {shlex.quote(self._container_path(generated))} && "
            f"cat >> {shlex.quote(self._container_path(generated))} <<'EOF'\n"
            + "\n".join(settings_append)
            + "\nEOF"
        )
        completed = self._container(["bash", "-lc", script], workdir="/work", no_display=True)
        if completed.returncode != 0:
            log = output_dir / "settings_generation.log"
            self._write_log(log, completed.stdout, completed.stderr)
            raise OrbSlam3RunError(
                "Failed to generate ORB-SLAM3 settings overlay.",
                returncode=completed.returncode,
                command=completed.args,
                log=log,
            )
        return generated

    def _ensure_pointcloud_helper(self, output_dir: Path) -> Path:
        helper = output_dir / "helpers" / "rgbd_keyframes_to_ply"
        if helper.exists():
            return helper
        helper.parent.mkdir(parents=True, exist_ok=True)
        source = "/work/tools/rgbd_keyframes_to_ply.cpp"
        log = helper.parent / "rgbd_keyframes_to_ply.build.log"
        completed = self._container(
            ["bash", "-lc", self._compile_cpp_command(source, self._container_path(helper), include_orbslam=False)],
            workdir="/work",
            no_display=True,
        )
        self._write_log(log, completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise OrbSlam3RunError(
                "Failed to compile rgbd_keyframes_to_ply.",
                returncode=completed.returncode,
                command=completed.args,
                log=log,
            )
        return helper

    def _ensure_observation_helper(self, output_dir: Path) -> Path:
        helper = output_dir / "helpers" / "sequence_observation_export"
        if helper.exists():
            return helper
        helper.parent.mkdir(parents=True, exist_ok=True)
        source = "/work/tools/sequence_observation_export.cpp"
        log = helper.parent / "sequence_observation_export.build.log"
        completed = self._container(
            ["bash", "-lc", self._compile_cpp_command(source, self._container_path(helper))],
            workdir="/work",
            no_display=True,
        )
        self._write_log(log, completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise OrbSlam3RunError(
                "Failed to compile sequence_observation_export.",
                returncode=completed.returncode,
                command=completed.args,
                log=log,
            )
        return helper

    def _compile_cpp_command(self, source: str, output: str, *, include_orbslam: bool = True) -> str:
        includes = [
            "-I/opt/orbslam3",
            "-I/opt/orbslam3/include",
            "-I/opt/orbslam3/include/CameraModels",
            "-I/opt/orbslam3/Thirdparty/Sophus",
            "-I/usr/include/eigen3",
        ]
        libs = [
            "-L/opt/orbslam3/lib",
            "-L/usr/local/lib",
            "-Wl,-rpath,/opt/orbslam3/lib",
            "-Wl,-rpath,/usr/local/lib",
        ]
        if include_orbslam:
            libs.extend(["-lORB_SLAM3", "-lpangolin", "-lOpenGL", "-lGLEW"])
        return " ".join(
            [
                "g++ -std=c++11 -O3 -DCOMPILEDWITHC11",
                shlex.quote(source),
                "-o",
                shlex.quote(output),
                *includes,
                "$(pkg-config --cflags opencv4)",
                *libs,
                "$(pkg-config --libs opencv4)",
                "-lpthread",
            ]
        )

    def _container(self, command: Sequence[str], *, workdir: str, no_display: bool):
        return run_container(
            command,
            ContainerOptions(
                image=self.image,
                workdir=workdir,
                mount_cwd=self.repo_root,
                no_display=no_display,
                interactive=False,
                env_vars={"LD_LIBRARY_PATH": DEFAULT_LD_LIBRARY_PATH},
            ),
            capture_output=True,
        )

    def _container_path(self, value: str | Path) -> str:
        if isinstance(value, str) and value.startswith("/"):
            return value
        path = Path(value)
        if path.is_absolute():
            if self._is_under_repo(path):
                return "/work/" + path.resolve().relative_to(self.repo_root).as_posix()
            if path.exists():
                raise ValueError(f"{path} is outside mounted repo root {self.repo_root}") from None
            return path.as_posix()
        return "/work/" + path.as_posix()

    def _is_under_repo(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.repo_root)
        except ValueError:
            return False
        return True

    def _atlas_paths(self, atlas: Path) -> tuple[str, Path]:
        atlas = Path(atlas)
        without_suffix = atlas.with_suffix("") if atlas.suffix == ".osa" else atlas
        final = without_suffix.with_suffix(".osa")
        if without_suffix.is_absolute():
            if self._is_under_repo(without_suffix):
                final.parent.mkdir(parents=True, exist_ok=True)
            final_host = final
        else:
            (self.repo_root / final.parent).mkdir(parents=True, exist_ok=True)
            final_host = self.repo_root / final
        return self._container_path(without_suffix), final_host

    @staticmethod
    def _write_log(path: Path, stdout: str, stderr: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if stdout:
                handle.write(stdout)
            if stderr:
                if stdout and not stdout.endswith("\n"):
                    handle.write("\n")
                handle.write(stderr)


class MonocularRunner(BaseRunner):
    sensor = "monocular"


class StereoRunner(BaseRunner):
    sensor = "stereo"
    manifest_columns = ("timestamp", "left", "right")

    def _manifest_row(self, frame: SequenceFrame, index: int) -> str:
        if not isinstance(frame, StereoFrame):
            raise ValueError(f"Stereo frame {index} must be a StereoFrame with left and right image paths.")
        return f"{float(frame.timestamp):.9f} {frame.left} {frame.right}\n"

    def _observation_command(
        self,
        *,
        executable: str | Path,
        dataset: Path,
        settings: str | Path,
        output_dir: Path,
        manifest: Path | None = None,
    ) -> list[str]:
        if manifest is None:
            raise ValueError("Stereo runs require frames=... or manifest=... with left and right image paths.")
        return super()._observation_command(
            executable=executable,
            dataset=dataset,
            settings=settings,
            output_dir=output_dir,
            manifest=manifest,
        )


class RGBDRunner(BaseRunner):
    sensor = "rgbd"
    manifest_columns = ("timestamp", "image", "depth")

    def __init__(
        self,
        *,
        image: str | None = None,
        vocabulary: str | Path = "/opt/orbslam3/Vocabulary/ORBvoc.txt",
        settings: str | Path,
        repo_root: Path | None = None,
    ) -> None:
        super().__init__(
            image=image,
            vocabulary=vocabulary,
            settings=settings,
            repo_root=repo_root,
        )

    def _manifest_row(self, frame: SequenceFrame, index: int) -> str:
        if isinstance(frame, RGBDFrame):
            return f"{float(frame.timestamp):.9f} {frame.image} {frame.depth}\n"
        if isinstance(frame, ImageFrame):
            if frame.depth is None:
                raise ValueError(f"RGB-D frame {index} is missing a depth path.")
            return f"{float(frame.timestamp):.9f} {frame.image} {frame.depth}\n"
        raise ValueError(f"RGB-D frame {index} must be an RGBDFrame or ImageFrame with a depth path.")

    def _observation_command(
        self,
        *,
        executable: str | Path,
        dataset: Path,
        settings: str | Path,
        output_dir: Path,
        manifest: Path | None = None,
    ) -> list[str]:
        command = [
            self._container_path(executable),
            self.sensor,
            self._container_path(self.vocabulary),
            self._container_path(settings),
            self._container_path(dataset),
            self._container_path(output_dir / "observations.jsonl"),
            self._container_path(output_dir / "map_points.csv"),
        ]
        if manifest is None:
            raise ValueError("RGB-D runs require frames=... or manifest=... with depth image paths.")
        command.extend(["--manifest", self._container_path(manifest)])
        return command
