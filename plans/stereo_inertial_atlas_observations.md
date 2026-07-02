# Stereo, Visual-Inertial, Atlas Loading, and Map-Posed Observations Plan

## Goal

Extend `pyorbslam3` from a monocular/RGB-D batch facade into a broader thin
wrapper for ORB-SLAM3's native C++ capabilities:

- Stereo SLAM.
- Monocular-inertial, stereo-inertial, and RGB-D-inertial SLAM.
- Atlas loading and localization against a saved map.
- Per-frame and per-keyframe posed observation artifacts, including ORB
  features and tracked map-point links, with poses clearly tied to either
  runtime tracking or the final optimized map.

This remains a thin Docker-backed wrapper. ORB-SLAM3 already implements the
SLAM algorithms in C++; this work adds narrow adapter executable support and
Python artifact plumbing, not a native Python binding layer.

## Current State

The current package is `pyorbslam3`.

Implemented today:

- `MonocularRunner`
- `RGBDRunner`
- generic batch sequence manifests
- map building via `System.SaveAtlasToFile`
- TUM trajectory parsing/writing
- observation export through `tools/sequence_observation_export.cpp`
- observed map-point CSV export
- RGB-D keyframe PLY generation

Important current limitations:

- `tools/sequence_observation_export.cpp` only accepts `monocular` and `rgbd`.
- There is no `StereoRunner`.
- There are no visual-inertial runners or IMU data models.
- `BaseRunner.localize()` intentionally raises `NotImplementedError`.
- Atlas saving exists, but atlas loading is not exposed.
- Observations captured immediately after `Track*()` are runtime tracking
  observations. They are not necessarily poses after loop closure or final
  bundle adjustment.

## Why Adapter C++ Is Needed

ORB-SLAM3 natively exposes the needed APIs:

- `ORB_SLAM3::System::TrackMonocular`
- `ORB_SLAM3::System::TrackStereo`
- `ORB_SLAM3::System::TrackRGBD`
- IMU vectors passed into the tracking calls
- `ActivateLocalizationMode`
- `DeactivateLocalizationMode`
- YAML-backed `System.SaveAtlasToFile`
- YAML-backed `System.LoadAtlasFromFile`
- `GetTrackingState`
- `GetTrackedKeyPointsUn`
- `GetTrackedMapPoints`

The stock ORB-SLAM3 example binaries are demo runners for specific datasets.
They do not provide the unified arbitrary-sequence manifest interface,
localization-mode workflow, or Python-friendly posed observation artifacts this
package needs.

The adapter executable should stay small: read manifests, call native
ORB-SLAM3 APIs, write durable artifacts.

## Public Python API Target

Add data models:

```python
@dataclass(frozen=True)
class ImageFrame:
    timestamp: float
    image: str | Path


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


@dataclass(frozen=True)
class IMUPoint:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
```

Keep the existing `ImageFrame(timestamp, image, depth=None)` compatibility if it
is practical, but new code should prefer explicit frame types.

Add runners:

```python
MonocularRunner
StereoRunner
RGBDRunner
MonocularInertialRunner
StereoInertialRunner
RGBDInertialRunner
```

Each runner should expose:

```python
run_sequence(...)
build_map(...)
localize(...)
```

The signatures can vary by sensor, but the common shape should stay familiar:

```python
result = slam.run_sequence(
    dataset=Path("datasets/sequence"),
    output_dir=Path("runs/sequence"),
    frames=[...],
    imu=[...],             # inertial runners only
    no_display=True,
)

mapping = slam.build_map(
    dataset=Path("datasets/map_run"),
    output_dir=Path("runs/map_run"),
    atlas=Path("maps/site_map"),
    frames=[...],
    imu=[...],
)

localized = slam.localize(
    dataset=Path("datasets/query_run"),
    output_dir=Path("runs/query_run"),
    atlas=Path("maps/site_map.osa"),
    frames=[...],
    imu=[...],
)
```

## Manifest Formats

Use explicit manifest files generated under `output_dir/configs/` when Python
receives dataclasses.

Monocular frames:

```text
timestamp image
```

Stereo frames:

```text
timestamp left_image right_image
```

RGB-D frames:

```text
timestamp image depth
```

IMU measurements:

```text
timestamp ax ay az gx gy gz
```

All image paths are relative to `dataset`. Keep comments and blank lines
allowed for hand-authored manifests.

## Sensor Mapping

Map Python runner classes to ORB-SLAM3 sensors:

```text
MonocularRunner          -> ORB_SLAM3::System::MONOCULAR
StereoRunner             -> ORB_SLAM3::System::STEREO
RGBDRunner               -> ORB_SLAM3::System::RGBD
MonocularInertialRunner  -> ORB_SLAM3::System::IMU_MONOCULAR
StereoInertialRunner     -> ORB_SLAM3::System::IMU_STEREO
RGBDInertialRunner       -> ORB_SLAM3::System::IMU_RGBD
```

The adapter executable should call:

```cpp
TrackMonocular(image, timestamp, imu_window)
TrackStereo(left, right, timestamp, imu_window)
TrackRGBD(image, depth, timestamp, imu_window)
```

For non-inertial runners, pass an empty IMU vector.

For inertial runners, collect IMU samples in the interval:

```text
previous_frame_timestamp < imu_timestamp <= current_frame_timestamp
```

For the first frame, follow the ORB-SLAM3 example convention for the matching
dataset style. Prefer including measurements up to the first frame timestamp,
with a small note in code comments if exact startup behavior differs by example.

## Adapter Executable

Evolve `tools/sequence_observation_export.cpp` or replace it with a cleaner
`tools/sequence_runner_export.cpp`.

Suggested CLI:

```text
sequence_runner_export \
  --sensor SENSOR \
  --vocab VOCAB \
  --settings SETTINGS \
  --dataset DATASET \
  --frames FRAMES_MANIFEST \
  --output-dir OUTPUT_DIR \
  [--imu IMU_MANIFEST] \
  [--mode slam|localize] \
  [--save-atlas ATLAS_BASENAME] \
  [--load-atlas ATLAS_BASENAME]
```

Keep compatibility with the current helper if easy, but prioritize a simpler
argument parser over preserving the old positional form internally.

Outputs:

```text
OUTPUT_DIR/
  orb_slam3.log
  CameraTrajectory.txt
  KeyFrameTrajectory.txt
  observations_runtime.jsonl
  observations_keyframes.jsonl
  map_points.csv
```

The Python wrapper can keep aliases:

- `observations_path` -> runtime observations for now.
- `map_observations_path` or `keyframe_observations_path` -> final map/keyframe
  observations.

## Atlas Loading And Localization

Map building:

- Generate a settings overlay with `System.SaveAtlasToFile`.
- Normalize atlas paths as the current `_atlas_paths()` does: accept `name` or
  `name.osa`, write the YAML value without `.osa`, and return the final `.osa`.

Localization:

- Generate a settings overlay with `System.LoadAtlasFromFile`.
- Construct `ORB_SLAM3::System` with the load setting present.
- Call `SLAM.ActivateLocalizationMode()` before processing frames.
- Do not save a new atlas unless the caller explicitly asks for it.
- Return poses in the loaded map frame.

The user-facing distinction should be:

```python
build_map(...)   # creates or updates map
localize(...)    # loads map and tracks query sequence against it
```

## Posed Observations Requirement

The package must make it possible to obtain a sequence of posed observations
after building a map:

- image paths
- optional right/depth image paths
- timestamp
- tracking state
- pose
- ORB keypoints
- tracked map-point IDs where available
- descriptor or descriptor reference if exported
- whether pose is runtime or final-map adjusted

Important semantic distinction:

Runtime observations are emitted immediately after each `Track*()` call. Their
poses are the live tracking poses returned by ORB-SLAM3 at that moment.

Final map/keyframe observations are emitted after `SLAM.Shutdown()`, using final
keyframe poses where available. Those poses are in the optimized map frame after
loop closure and bundle adjustment.

Do not silently mix those meanings. Add a field such as:

```json
{
  "pose_basis": "runtime_tracking"
}
```

or:

```json
{
  "pose_basis": "final_map_keyframe"
}
```

For non-keyframe frames, if final adjusted poses are not available, either omit
them from the final-map artifact or include them with:

```json
{
  "pose_basis": "runtime_tracking",
  "final_map_adjusted": false
}
```

Prefer correctness and explicit metadata over pretending every frame has a
post-optimization pose.

## Artifact Parser Updates

Extend `pyorbslam3.artifacts` with:

```python
PoseBasis = Literal["runtime_tracking", "final_map_keyframe", "runtime_tracking_in_loaded_map"]
```

or a simple string field if avoiding `typing.Literal`.

Extend `PosedObservation` to include:

- `sensor`
- `pose_basis`
- `left`, `right`, `image`, and `depth` fields as appropriate
- optional IMU sample count for inertial runs

Keep backward compatibility with existing `image` and `depth` fields when
possible.

## Implementation Phases

### Phase 1: Refactor Frame Modeling

- Add `StereoFrame`, `RGBDFrame`, and `IMUPoint`.
- Keep current `ImageFrame` behavior working.
- Move manifest generation behind sensor-specific methods.
- Add unit tests for manifest rows and validation errors.

Acceptance criteria:

- Existing monocular and RGB-D tests still pass.
- Missing depth/right image paths produce clear Python errors.
- IMU data is accepted only by inertial runners.

### Phase 2: Stereo Runner

- Add `StereoRunner`.
- Teach the C++ helper to read stereo manifests.
- Call `TrackStereo`.
- Export runtime observations and map points.

Acceptance criteria:

- `StereoRunner.run_sequence()` compiles/runs the helper.
- It writes `KeyFrameTrajectory.txt`, `observations_runtime.jsonl`, and
  `map_points.csv`.
- Unsupported old sensor strings fail with clear messages.

### Phase 3: Shared IMU Plumbing

- Add IMU manifest generation.
- Add C++ IMU manifest reader.
- Add timestamp-window assignment.
- Pass IMU vectors to `TrackMonocular`, `TrackStereo`, or `TrackRGBD`.

Acceptance criteria:

- IMU samples are sorted or validated as monotonic.
- Empty IMU input for an inertial runner fails before invoking ORB-SLAM3.
- C++ logs include frame count and IMU sample count.

### Phase 4: Inertial Runners

- Add `MonocularInertialRunner`.
- Add `StereoInertialRunner`.
- Add `RGBDInertialRunner`.

Acceptance criteria:

- All three runners use the same helper and artifact path conventions.
- They select the correct ORB-SLAM3 sensor enum.
- They produce runtime posed observations.

### Phase 5: Atlas Loading And Localization

- Implement `localize(...)`.
- Add `System.LoadAtlasFromFile` settings overlay.
- Add helper `--mode localize`.
- Call `ActivateLocalizationMode`.

Acceptance criteria:

- `build_map(...).atlas` points to an existing `.osa`.
- `localize(..., atlas=that_atlas)` loads the map and returns query poses.
- Result metadata distinguishes `mode="slam"` from `mode="localize"`.

### Phase 6: Final Map/Keyframe Observations

- After `SLAM.Shutdown()`, export final keyframe poses and observations where
  available.
- Add parser support for the new artifact.
- Return it on `RunResult`.

Acceptance criteria:

- Map-building results include runtime observations and final-map/keyframe
  observations.
- Each observation includes `pose_basis`.
- Documentation explains why runtime frame poses and final keyframe poses can
  differ after loop closure.

### Phase 7: Docs And Examples

- Update `docs/thin_python_wrapper.md`.
- Add example snippets for stereo, mono-inertial, stereo-inertial,
  RGB-D-inertial, map building, and localization.
- Keep a short explanation of why adapter C++ exists.

Acceptance criteria:

- A new user can identify the right runner class for each ORB-SLAM3 sensor mode.
- The map-posed observation semantics are documented plainly.

## Testing Strategy

Unit tests not requiring Docker:

- Docker command construction.
- Atlas path normalization.
- Manifest generation and validation.
- Artifact parsing.
- Observation parser backward compatibility.

Docker/integration tests when available:

- Compile helper in the image.
- Run a very small monocular sequence.
- Run stereo against a tiny fixture or documented sample if available.
- Verify atlas save path exists after `build_map`.
- Verify `localize` invokes the helper with load-atlas and localization mode.

Do not require large public datasets in the default unit test suite.

## Open Questions

- Should final map observations include only keyframes, or should the wrapper
  attempt to post-align non-keyframe runtime poses to the final map? Prefer
  keyframes only at first unless ORB-SLAM3 exposes a reliable final frame pose
  source.
- Should descriptors be embedded in every observation, or should observations
  reference map-point descriptors from `map_points.csv`? Prefer references to
  keep JSONL sizes reasonable.
- Should old `ImageFrame(depth=...)` remain the preferred RGB-D input or become
  a compatibility path? Prefer explicit `RGBDFrame` in docs.
