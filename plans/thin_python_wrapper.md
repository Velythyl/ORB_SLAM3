# Thin Python Wrapper Plan

## Goal

Build a small Python API that feels like ORB-SLAM3 bindings for common workflows,
but avoids native Python bindings entirely. Python should only orchestrate the
existing Docker-backed ORB-SLAM3 image, compile or call narrow no-viewer helper
executables, and parse the files those executables produce.

This is intentionally not a `pybind11` or C-extension project. The wrapper is a
friendly shell around container commands and durable artifacts.

## Why This Approach

Real Python bindings would need to expose `ORB_SLAM3::System`, OpenCV image
types, Sophus poses, atlas lifecycle methods, localization mode, threading
shutdown semantics, and probably callback-style frame ingestion. That is a lot
of ABI and ownership surface for a codebase that already works well as compiled
C++ inside a container.

The thin wrapper gives us most of the user-facing value sooner:

- Python users can call `slam.run_tum(...)`, `slam.build_map(...)`, and
  `slam.localize(...)` from scripts and notebooks.
- ORB-SLAM3 remains built and executed inside the known-good container image.
- Outputs stay inspectable: TUM trajectory files, `.osa` atlas files, logs, and
  optional PLY point clouds.
- The CLI and Python package share the same Docker runner instead of drifting
  into separate systems.

## Existing Context

Current package:

```text
src/orb_slam3_runner/
```

Current console entry point:

```text
orb-slam3 = orb_slam3_runner.cli:main
```

Current Docker image defaults:

```text
ghcr.io/velythyl/orb_slam3:latest
ORB_SLAM3_IMAGE=orb-slam3:local
```

The workflow in `docs/trajectory_pointcloud_and_localization.md` already proves:

- Docker-backed ORB-SLAM3 execution works.
- No-viewer executables are needed for headless runs.
- `KeyFrameTrajectory.txt` can be produced and parsed.
- RGB-D plus keyframe poses can generate a colored PLY.
- Atlas save/load and localization mode are available through ORB-SLAM3 C++ APIs
  and YAML fields.

## Design Principle

Expose Python objects for ergonomics, but make every operation map to one of
these actions:

1. Prepare files in a run directory.
2. Run a container command.
3. Read output files.
4. Return lightweight Python frozen dataclasses.

Do not try to pass live frames into a native ORB-SLAM3 object from Python for
the first version.

## Proposed Python API

Package namespace:

```python
import orb_slam3
```

Core objects:

```python
from pathlib import Path
from orb_slam3 import MonocularRunner, RGBDRunner

slam = MonocularRunner(
    image="orb-slam3:local",
    vocabulary="/opt/orbslam3/Vocabulary/ORBvoc.txt",
    settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
)

result = slam.run_tum(
    dataset=Path("datasets/rgbd_dataset_freiburg1_xyz"),
    output_dir=Path("runs/freiburg1_xyz"),
    no_display=True,
)

print(result.keyframe_trajectory)
print(result.poses[:3])
```

Map and localization API:

```python
mapping = slam.build_map(
    dataset=Path("datasets/rgbd_dataset_freiburg1_xyz"),
    atlas=Path("maps/freiburg1_xyz_map"),
    output_dir=Path("runs/freiburg1_xyz_mapping"),
)

localized = slam.localize(
    dataset=Path("datasets/rgbd_dataset_freiburg1_xyz"),
    atlas=Path("maps/freiburg1_xyz_map.osa"),
    output_dir=Path("runs/query_300_500"),
    start_frame=300,
    end_frame=500,
)
```

Point cloud API:

```python
ply = result.make_pointcloud(
    output=Path("pointclouds/freiburg1_xyz_orb_keyframes.ply"),
)
```

The first implementation can support only batch TUM-style datasets. Live camera
streaming can come later as a separate process protocol.

## Data Model

Add small dataclasses in `src/orb_slam3_runner/api.py` or rename the importable
package to `orb_slam3` while keeping CLI compatibility.

Suggested dataclasses:

```python
@dataclass(frozen=True)
class Pose:
    timestamp: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    keyframe_trajectory: Path | None
    camera_trajectory: Path | None
    atlas: Path | None
    log: Path
    poses: tuple[Pose, ...]
```

Keep these plain and dependency-free. Do not require `numpy` for the base
package.

## Implementation Phases

### Phase 1: Extract Docker Runner Internals

Move the reusable parts of `src/orb_slam3_runner/cli.py` into a module such as:

```text
src/orb_slam3_runner/docker.py
```

Expose:

```python
run_container(command: Sequence[str], options: ContainerOptions) -> CompletedRun
```

The CLI should call the same function. The Python API should call it too.

Acceptance criteria:

- Existing `orb-slam3 image`, `orb-slam3 pull`, `orb-slam3 run`, and
  `orb-slam3 shell` behavior remains unchanged.
- Unit tests can verify Docker command construction without needing Docker.

### Phase 2: Add Artifact Parsers

Add parsers for TUM trajectory files:

```text
timestamp tx ty tz qx qy qz qw
```

Expose:

```python
read_tum_trajectory(path: Path) -> tuple[Pose, ...]
write_tum_trajectory(path: Path, poses: Iterable[Pose]) -> None
```

Acceptance criteria:

- Empty/comment lines are ignored.
- Bad rows produce clear exceptions with line numbers.
- Parser works for both `KeyFrameTrajectory.txt` and camera trajectory output.

### Phase 3: Add Batch Dataset Runners

Implement `MonocularRunner.run_tum(...)` as the first wrapper method.

It should:

1. Create an output directory.
2. Run the no-viewer executable in Docker.
3. Capture stdout/stderr to a log file.
4. Move or copy `KeyFrameTrajectory.txt` into the output directory.
5. Parse poses and return `RunResult`.

For the initial version, require the caller to provide the executable path if it
is not available in the image:

```python
MonocularRunner(..., executable="/work/datasets/mono_tum_noviewer")
```

Later, add a helper that compiles the no-viewer executable if missing.

Acceptance criteria:

- The Python call can reproduce the documented Freiburg run.
- `RunResult.keyframe_trajectory` points to the saved file under `output_dir`.
- Failed container runs include the log path in the raised exception.

### Phase 4: Add Map Save/Load Helpers

The wrapper should generate temporary YAML settings files instead of forcing the
user to hand-edit configs.

For map building:

```yaml
System.SaveAtlasToFile: "maps/freiburg1_xyz_map"
```

For localization:

```yaml
System.LoadAtlasFromFile: "maps/freiburg1_xyz_map"
```

Because ORB-SLAM3 appends `.osa`, normalize atlas paths carefully:

- Accept `maps/name` or `maps/name.osa`.
- Write YAML value without `.osa`.
- Return the final `.osa` path.

Acceptance criteria:

- `build_map(...)` creates an atlas file in the expected host path.
- Generated YAML files are stored under `output_dir/configs/` for debugging.
- The wrapper documents that relative paths are resolved inside `/work`.

### Phase 5: Add Localization Wrapper Executable

The Python shell still needs one narrow C++ helper for localization mode,
because stock examples do not call `ActivateLocalizationMode()` or expose frame
range options.

Add a C++ runner such as:

```text
tools/mono_tum_localize.cc
tools/rgbd_tum_localize.cc
```

Required options:

```text
--start-frame N
--end-frame N
--save-frame-trajectory FILE
--save-keyframe-trajectory FILE
--activate-localization
```

The Python API should compile this helper inside Docker when needed, then call
it as a normal container command.

Acceptance criteria:

- `localize(...)` loads an existing atlas, activates localization mode, and
  writes camera poses for the query segment.
- Returned poses are in the loaded map frame.
- The first supported sensor mode can be monocular, but the plan should keep
  room for RGB-D because metric robot localization needs scale.

### Phase 6: Add Point Cloud Convenience

Wrap `tools/rgbd_keyframes_to_ply.cpp` as an optional convenience method:

```python
result.make_pointcloud(dataset=..., output=...)
```

The method should compile the converter inside Docker if the binary is missing,
then call it with:

```text
dataset
KeyFrameTrajectory.txt
output.ply
```

Acceptance criteria:

- Produces the same binary little-endian PLY as the documented workflow.
- Validates the PLY header vertex count after generation.
- Warns clearly when using monocular poses with RGB-D depth because scale is
  ambiguous.

## Package Layout

Recommended first cut:

```text
src/orb_slam3_runner/
  __init__.py
  api.py
  artifacts.py
  cli.py
  docker.py
  errors.py
  paths.py
```

Optional compatibility alias later:

```text
src/orb_slam3/
  __init__.py
```

That alias can re-export the public API while `orb_slam3_runner` remains the
implementation package used by the CLI.

## Error Handling

Define one main exception:

```python
class OrbSlam3RunError(RuntimeError):
    returncode: int
    command: tuple[str, ...]
    log: Path | None
```

When Docker fails, raise this instead of returning a bare non-zero code from the
Python API. The CLI can still return process-style exit codes.

## Testing Strategy

Fast tests:

- Docker command construction.
- Mount path normalization.
- Atlas path normalization.
- YAML overlay generation.
- TUM trajectory parser.
- Error messages for bad trajectory rows.

Integration tests:

- Mark as Docker-required.
- Run a tiny command in the configured image, such as `help`.
- If the Freiburg dataset and no-viewer binary exist, run a short frame range
  smoke test.

Do not make network downloads part of normal tests.

## Documentation

Add a short user-facing doc:

```text
docs/thin_python_wrapper.md
```

It should show:

- Install from the local wheel or source checkout.
- Override image with `ORB_SLAM3_IMAGE=orb-slam3:local`.
- Run a TUM dataset from Python.
- Build an atlas.
- Localize a query segment.
- Generate a PLY.

Be explicit that this is a container-backed facade, not native in-process
bindings.

## Out of Scope for Version 1

- Native Python extension modules.
- In-process frame-by-frame ORB-SLAM3 calls.
- Real-time camera capture from Python.
- ROS integration.
- A stable binary ABI.
- Automatic dataset downloads.
- Full `numpy` pose/matrix API.

## Open Questions

- Should the public import name be `orb_slam3` immediately, or should version 1
  keep `orb_slam3_runner` and add the nicer alias later? we can fix later
- Should helper executables be built into the Docker image, compiled on demand,
  or both? Go with the cleanest design you can think of. Maybe we need to modify the C++ files to expose hooks to automatically detect data structures etc. You have full control, do what you think is cleanest & most maintainable
- Which metric mode should be first-class for robot localization: RGB-D or
  stereo-inertial? RGB-D, but we want to support all modes
- Should live robot use be handled by a long-running container process with a
  small IPC protocol, rather than repeated batch runs? Long-running process

## Recommended Next Step

Implement Phases 1 through 3 first. That gives Python users a credible wrapper
for the already-proven trajectory workflow without touching ORB-SLAM3 internals.
Then add the localization helper executable and atlas YAML generation as the
next focused milestone.
