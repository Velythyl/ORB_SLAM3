# Python Client Guide

`pyorbslam3` is a Docker-backed Python client for this ORB-SLAM3 fork. It is
not a native binding to `ORB_SLAM3::System`: Python prepares manifests and
settings, runs helper binaries inside the ORB-SLAM3 container, captures logs,
and parses the generated trajectory, observation, map point, atlas, and PLY
artifacts.

Use this document as the main reference when writing Python code against this
package.

## Codebase Map

Start here when changing the Python client:

```text
pyorbslam3/api.py          high-level runners, manifest generation, atlas paths
pyorbslam3/artifacts.py    dataclasses and readers for trajectories, JSONL, CSV, PLY
pyorbslam3/docker.py       Docker command construction and execution
pyorbslam3/cli.py          orb-slam3 command line entry point
pyorbslam3/errors.py       OrbSlam3RunError
tools/sequence_observation_export.cpp
                           C++ helper for monocular/stereo/RGB-D runs and observations
tools/rgbd_keyframes_to_ply.cpp
                           C++ helper for RGB-D point cloud generation
tests/test_api.py          runner path, manifest, and command behavior
tests/test_artifacts.py    artifact parser behavior
tests/test_docker.py       Docker command construction
```

The Python runner compiles helper binaries into `output_dir/helpers/` on demand.
Those helpers are executed inside Docker with the repository mounted at `/work`.

## Install

From this checkout:

```bash
python -m pip install -e .
```

With `uv`:

```bash
uv sync --extra test --extra viewer
```

The default Docker image is:

```text
ghcr.io/velythyl/orb_slam3:latest
```

Override it for local development:

```bash
export ORB_SLAM3_IMAGE=orb-slam3:local
```

The package assumes Docker is installed and that the project root is mounted at
`/work` inside the container. The built ORB-SLAM3 tree in the image is expected
at `/opt/orbslam3`.

## Quick Start

Run a timestamped monocular sequence:

```python
from pathlib import Path

from pyorbslam3 import ImageFrame, MonocularRunner

slam = MonocularRunner(
    settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
)

result = slam.run_sequence(
    dataset=Path("datasets/my_sequence"),
    output_dir=Path("runs/my_sequence"),
    frames=[
        ImageFrame(0.000000, "rgb/frame_000001.png"),
        ImageFrame(0.033333, "rgb/frame_000002.png"),
    ],
)

print(result.keyframe_trajectory)
print(len(result.poses))
print(result.observations_path)
print(result.map_points_path)
print(result.log)
```

`run_dataset(...)` and `run_sequence(...)` are aliases. Use either one for
explicit timestamped image sequences.

## Runner Classes

Import runners from `pyorbslam3`:

```python
from pyorbslam3 import MonocularRunner, RGBDRunner, StereoRunner
```

Each runner accepts:

```python
runner = MonocularRunner(
    image=None,  # optional Docker image override
    vocabulary="/opt/orbslam3/Vocabulary/ORBvoc.txt",
    settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
    repo_root=None,  # defaults to Path.cwd()
)
```

Use a host path under `repo_root` for datasets and outputs. Relative paths are
interpreted under `repo_root` and mapped to `/work/...` in the container.
Absolute paths outside `repo_root` are only accepted when they are intended to
already exist inside the container, such as `/opt/orbslam3/...`.

### MonocularRunner

Use `ImageFrame(timestamp, image)` and a manifest with two columns:

```text
timestamp image
```

Example:

```python
from pathlib import Path

from pyorbslam3 import ImageFrame, MonocularRunner

slam = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml")
result = slam.run_sequence(
    dataset=Path("datasets/tum_rgbd"),
    output_dir=Path("runs/tum_mono"),
    frames=[
        ImageFrame(1305031102.175304, "rgb/1305031102.175304.png"),
    ],
)
```

### StereoRunner

Use `StereoFrame(timestamp, left, right)` and a manifest with three columns:

```text
timestamp left_image right_image
```

Example:

```python
from pathlib import Path

from pyorbslam3 import StereoFrame, StereoRunner

stereo = StereoRunner(settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml")
result = stereo.run_sequence(
    dataset=Path("datasets/euroc/MH01"),
    output_dir=Path("runs/euroc_mh01"),
    frames=[
        StereoFrame(
            1403636579.764555,
            "mav0/cam0/data/1403636579764555584.png",
            "mav0/cam1/data/1403636579764555584.png",
        ),
    ],
)
```

Stereo runs require `frames=...` or `manifest=...`; there is no implicit dataset
scanner.

### RGBDRunner

Use `RGBDFrame(timestamp, image, depth)` or the compatible
`ImageFrame(timestamp, image, depth=...)` form. The manifest has three columns:

```text
timestamp image depth
```

Example:

```python
from pathlib import Path

from pyorbslam3 import RGBDFrame, RGBDRunner

rgbd = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml")
result = rgbd.run_sequence(
    dataset=Path("datasets/rgbd_dataset_freiburg1_xyz"),
    output_dir=Path("runs/freiburg1_xyz_rgbd"),
    frames=[
        RGBDFrame(
            1305031102.175304,
            "rgb/1305031102.175304.png",
            "depth/1305031102.160407.png",
        ),
    ],
)
```

RGB-D runs require a depth path for every frame.

## Manifests

Pass either `frames=...` or `manifest=...`, not both.

When you pass `frames`, the client writes:

```text
output_dir/configs/sequence_manifest.txt
```

and stores that path on:

```python
result.sequence_manifest
```

When you pass `manifest`, paths inside the manifest must be relative to
`dataset`. Absolute frame paths can be used in `frames`, but only when `dataset`
is also supplied and the frame path is under that dataset root; the generated
manifest will contain the relative path.

## RunResult

Runner methods return a `RunResult`:

```python
result.output_dir
result.keyframe_trajectory
result.camera_trajectory
result.atlas
result.observations_path
result.map_points_path
result.log
result.poses
result.observations
result.map_points
result.dataset
result.sensor
result.sequence_manifest
```

`poses` are parsed from `KeyFrameTrajectory.txt` when present, otherwise from
`CameraTrajectory.txt` when present. The trajectory format is TUM:

```text
timestamp tx ty tz qx qy qz qw
```

`observations` and `map_points` are parsed from helper exports generated for the
run:

```text
observations.jsonl
map_points.csv
```

These are inspectable runtime exports. The reusable ORB-SLAM3 map artifact is
the atlas `.osa` produced by `build_map(...)`.

## Build an Atlas

Use `build_map(...)` to run a sequence and ask ORB-SLAM3 to save an atlas:

```python
from pathlib import Path

mapping = slam.build_map(
    dataset=Path("datasets/my_sequence"),
    atlas=Path("maps/my_sequence_map"),
    output_dir=Path("runs/my_sequence_mapping"),
    manifest=Path("configs/my_sequence_manifest.txt"),
)

print(mapping.atlas)  # maps/my_sequence_map.osa
```

`build_map_sequence(...)` is an alias. The atlas path may be supplied with or
without `.osa`; ORB-SLAM3 appends `.osa`, so the client normalizes the final
host path for you.

Localization mode is not implemented yet. Calling `localize(...)` raises
`NotImplementedError`.

## Point Clouds

Generate a colored PLY from RGB-D data and keyframe poses:

```python
from pathlib import Path

ply = result.make_pointcloud(
    output=Path("pointclouds/my_sequence_keyframes.ply"),
)
```

If the original run used a sequence manifest, `RunResult.make_pointcloud(...)`
reuses it automatically. For custom data, pass camera intrinsics and depth
scale:

```python
from pathlib import Path

from pyorbslam3 import RGBDCamera

ply = result.make_pointcloud(
    output=Path("pointclouds/room0.ply"),
    camera=RGBDCamera(
        fx=600.0,
        fy=600.0,
        cx=599.5,
        cy=339.5,
        depth_scale=6553.5,
    ),
    stride=6,
    min_depth=0.25,
    max_depth=4.5,
)
```

Read a PLY:

```python
cloud = result.read_pointcloud(ply)
print(cloud.points.shape)
print(cloud.colors.shape if cloud.colors is not None else None)
```

View with Open3D:

```python
from pyorbslam3 import view_pointcloud

view_pointcloud(ply)
# or
result.view_pointcloud(ply)
```

Install the viewer dependencies first:

```bash
uv sync --extra viewer
```

## Replica room0 Example

The Replica example used the local Docker image because the default GHCR image
does not publish an Apple Silicon manifest. Make sure Docker is running, then
run from the repository root:

```bash
export ORB_SLAM3_IMAGE=orb-slam3:local
```

Run the RGB-D sequence with the checked-in Replica settings and manifest:

```bash
uv run python -c '
from pathlib import Path
from pyorbslam3 import RGBDRunner

slam = RGBDRunner(
    image="orb-slam3:local",
    settings=Path("runs/replica_room0_smoke/configs/replica_rgbd.yaml"),
    repo_root=Path.cwd(),
)
result = slam.run_sequence(
    dataset=Path("datasets/Replica/room0"),
    output_dir=Path("runs/replica_room0_rgbd"),
    manifest=Path("runs/replica_room0_rgbd/configs/sequence_manifest.txt"),
    no_display=True,
)
print(f"poses={len(result.poses)}")
print(f"observations={len(result.observations)}")
print(f"map_points={len(result.map_points)}")
print(f"log={result.log}")
'
```

Generate the colored keyframe point cloud:

```bash
uv run python -c '
from pathlib import Path
from pyorbslam3 import RGBDCamera, RGBDRunner
from pyorbslam3.artifacts import read_ply_vertex_count

slam = RGBDRunner(
    image="orb-slam3:local",
    settings=Path("runs/replica_room0_smoke/configs/replica_rgbd.yaml"),
    repo_root=Path.cwd(),
)
ply = slam.make_pointcloud(
    dataset=Path("datasets/Replica/room0"),
    trajectory=Path("runs/replica_room0_rgbd/KeyFrameTrajectory.txt"),
    output=Path("pointclouds/replica_room0_rgbd_keyframes.ply"),
    manifest=Path("runs/replica_room0_rgbd/configs/sequence_manifest.txt"),
    camera=RGBDCamera(fx=600.0, fy=600.0, cx=599.5, cy=339.5, depth_scale=6553.5),
    stride=4,
    min_depth=0.25,
    max_depth=4.5,
)
print(f"ply={ply}")
print(f"vertices={read_ply_vertex_count(ply)}")
'
```

Open the 3D map visualization:

```bash
uv run python -c '
from pathlib import Path
from pyorbslam3 import view_pointcloud

view_pointcloud(
    Path("pointclouds/replica_room0_rgbd_keyframes.ply"),
    window_name="Replica room0 ORB-SLAM3 Map",
)
'
```

On the last known run, this produced 31 keyframe poses, 120 observations,
16,013 exported ORB-SLAM3 map points, and a 1,465,750 vertex PLY.

## Artifact Helpers

These helpers are exported from `pyorbslam3`:

```python
from pyorbslam3 import (
    point_positions_to_numpy,
    poses_to_matrices,
    poses_to_numpy,
    read_map_points_csv,
    read_observations_jsonl,
    read_ply_pointcloud,
    read_tum_trajectory,
    to_open3d_pointcloud,
    view_pointcloud,
    write_tum_trajectory,
)
```

Common conversions:

```python
from pathlib import Path

poses = read_tum_trajectory(Path("runs/out/KeyFrameTrajectory.txt"))
pose_array = poses_to_numpy(poses)      # shape: (N, 8)
pose_mats = poses_to_matrices(poses)    # shape: (N, 4, 4)

points = read_map_points_csv(Path("runs/out/map_points.csv"))
xyz = point_positions_to_numpy(points)  # shape: (N, 3)
```

`numpy` is imported lazily. Install it before using NumPy conversion helpers.
`open3d` is imported lazily for point cloud viewing.

## Data Classes

Important public data classes:

```python
ImageFrame(timestamp: float, image: str | Path, depth: str | Path | None = None)
StereoFrame(timestamp: float, left: str | Path, right: str | Path)
RGBDFrame(timestamp: float, image: str | Path, depth: str | Path)
IMUPoint(timestamp: float, ax: float, ay: float, az: float, gx: float, gy: float, gz: float)
RGBDCamera(fx: float, fy: float, cx: float, cy: float, depth_scale: float = 5000.0)
Pose(timestamp: float, tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float)
MapPoint(id: int, x: float, y: float, z: float, observations: int, found: int, descriptor_hex: str = "")
FeatureObservation(...)
PosedObservation(...)
PointCloud(points, colors=None)
```

`Pose` has:

```python
pose.as_tuple()
pose.as_matrix()
```

`pose.as_matrix()` returns a NumPy `4x4` transform.

`IMUPoint` is currently only a public data container. The current runner methods
do not consume IMU samples.

## Docker Helpers

For lower-level container use:

```python
from pyorbslam3.docker import ContainerOptions, run_container

completed = run_container(
    ["mono_tum", "--help"],
    ContainerOptions(
        image="orb-slam3:local",
        no_display=True,
        mounts=("/host/data:/data:ro",),
        devices=(),
        env=("MY_FLAG=1",),
        gpu=None,
    ),
    capture_output=True,
)

print(completed.returncode)
print(completed.stdout)
print(completed.stderr)
```

The CLI uses these helpers internally.

## CLI

The package installs `orb-slam3`:

```bash
orb-slam3 image
orb-slam3 pull
orb-slam3 shell
orb-slam3 run -- mono_tum \
  /opt/orbslam3/Vocabulary/ORBvoc.txt \
  /opt/orbslam3/Examples/Monocular/TUM1.yaml \
  /work/datasets/rgbd_dataset_freiburg1_xyz
```

Useful options for `run` and `shell`:

```text
--image IMAGE
--workdir /work
--mount /host:/container:ro
--device /dev/video0
--env KEY=VALUE
--gpu all
--name NAME
--user 1000:1000
--privileged
--no-display
```

Use `--no-display` for headless runs.

## Errors and Logs

Container failures raise `OrbSlam3RunError`:

```python
from pyorbslam3 import OrbSlam3RunError

try:
    result = slam.run_sequence(...)
except OrbSlam3RunError as exc:
    print(exc.returncode)
    print(exc.command)
    print(exc.log)
```

Every high-level run writes logs under the output directory, usually:

```text
output_dir/orb_slam3.log
output_dir/helpers/sequence_observation_export.build.log
```

Point cloud generation writes a sibling log next to the requested PLY:

```text
pointclouds/my_sequence_keyframes.ply.log
```

## Current Limitations

- The client is container-backed, not an in-process Python extension.
- `localize(...)` is intentionally not implemented yet.
- Runner methods require explicit `frames=...` or `manifest=...`; they do not
  scan arbitrary dataset layouts.
- Stereo and RGB-D runs require manifests because the helper needs paired image
  paths.
- `IMUPoint` is exported but not wired into the runner APIs.
- NumPy and Open3D are optional dependencies and are loaded only by conversion
  and viewer helpers.
