# Thin Python Wrapper

This package is a container-backed facade around ORB-SLAM3, not native Python
bindings. Python prepares files, runs the configured Docker image, captures logs,
and parses durable artifacts such as TUM trajectories, atlas files, and PLY point
clouds.

Install from this checkout:

```bash
python -m pip install -e .
```

Use the local image built for this repository:

```bash
export ORB_SLAM3_IMAGE=orb-slam3:local
```

Run a monocular sequence with an explicit timestamp manifest:

```python
from pathlib import Path
from pyorbslam3 import ImageFrame, MonocularRunner

slam = MonocularRunner(
    settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
)

result = slam.run_dataset(
    dataset=Path("datasets/my_sequence"),
    output_dir=Path("runs/my_sequence"),
    frames=[
        ImageFrame(0.000000, "rgb/frame_000001.png"),
        ImageFrame(0.033333, "rgb/frame_000002.png"),
    ],
)

print(result.keyframe_trajectory)
print(result.poses[:3])
print(result.observations_path)
print(result.map_points_path)
print(result.log)
```

You can also use `run_sequence(...)`, which is an alias for explicit timestamped
sequence runs. The manifest is whitespace-separated, with paths relative to
`dataset`:

```text
0.000000 rgb/frame_000001.png
0.033333 rgb/frame_000002.png
```

```python
from pyorbslam3 import ImageFrame

result = slam.run_sequence(
    dataset=Path("datasets/my_sequence"),
    output_dir=Path("runs/my_sequence"),
    frames=[
        ImageFrame(0.000000, "rgb/frame_000001.png"),
        ImageFrame(0.033333, "rgb/frame_000002.png"),
    ],
)
```

For RGB-D, prefer `RGBDFrame`, or keep using the compatible
`ImageFrame(..., depth=...)` form:

```text
0.000000 rgb/frame_000001.png depth/frame_000001.png
0.033333 rgb/frame_000002.png depth/frame_000002.png
```

The generated manifest is recorded on the result:

```python
print(result.sequence_manifest)
```

Stereo uses `StereoFrame` and a manifest with left and right image paths:

```python
from pyorbslam3 import StereoFrame, StereoRunner

stereo = StereoRunner(
    settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml",
)

result = stereo.run_sequence(
    dataset=Path("datasets/euroc/MH01"),
    output_dir=Path("runs/euroc_mh01"),
    frames=[
        StereoFrame(1403636579.764555, "mav0/cam0/data/1403636579764555584.png", "mav0/cam1/data/1403636579764555584.png"),
    ],
)
```

The stereo manifest format is:

```text
timestamp left_image right_image
```

The wrapper compiles `tools/sequence_observation_export.cpp` into
`output_dir/helpers/` inside the container and reuses it on later calls.

Build an atlas from the same explicit sequence:

```python
mapping = slam.build_map(
    dataset=Path("datasets/my_sequence"),
    atlas=Path("maps/my_sequence_map"),
    output_dir=Path("runs/my_sequence_mapping"),
    manifest=Path("configs/my_sequence_manifest.txt"),
)

print(mapping.atlas)  # maps/my_sequence_map.osa
```

`build_map_sequence(...)` is an alias that accepts the same `frames=` or
`manifest=` arguments.

The generated settings file is stored under `output_dir/configs/` for debugging.
Atlas YAML values are written without `.osa` because ORB-SLAM3 appends the suffix
itself. Relative paths are resolved inside the repository mount at `/work`.

Generate a colored PLY from RGB-D frames and keyframe poses:

```python
ply = result.make_pointcloud(
    output=Path("pointclouds/my_sequence_keyframes.ply"),
)
```

For arbitrary RGB-D datasets, pass camera intrinsics and depth scale. If the run
used `run_sequence(...)`, the same manifest is reused automatically:

```python
from pyorbslam3 import RGBDCamera

ply = result.make_pointcloud(
    output=Path("pointclouds/replica_room0.ply"),
    camera=RGBDCamera(fx=600.0, fy=600.0, cx=599.5, cy=339.5, depth_scale=6553.5),
    stride=6,
)
```

Read the PLY back as optional NumPy arrays:

```python
cloud = result.read_pointcloud(ply)
print(cloud.points.shape)  # (N, 3)
print(cloud.colors.shape)  # (N, 3), when RGB is present
```

View it with Open3D by installing the optional viewer extra:

```bash
uv sync --extra viewer
```

```python
from pyorbslam3 import view_pointcloud

view_pointcloud(ply)
# or: result.view_pointcloud(ply)
```

For every run, the wrapper compiles and runs
`tools/sequence_observation_export.cpp`. It records:

- `observations.jsonl`: one posed frame per line, including image path, optional
  stereo right path, optional depth path, tracking state, `Twc` pose when
  tracking is valid, and extracted keypoints with linked map point ids where
  available. These are runtime tracking observations captured immediately after
  `Track*()`, not final optimized map/keyframe poses after loop closure.
- `map_points.csv`: deduplicated map points observed by tracked features,
  including world position, ORB-SLAM3 observation/found counts, and the
  representative descriptor as hex.

Those files are parsed into:

```python
result.observations
result.map_points
```

The authoritative reusable ORB-SLAM3 map is still the atlas `.osa` produced by
`build_map(...)`. The CSV map points are an inspectable export of points observed
during the run, suitable for Python analysis and visualization.

`RGBDRunner` uses the same explicit sequence format. Every frame needs both an
RGB image path and a depth image path:

```python
from pyorbslam3 import RGBDRunner

rgbd = RGBDRunner(
    settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml",
)

result = rgbd.run_dataset(
    dataset=Path("datasets/replica_room0"),
    output_dir=Path("runs/replica_room0"),
    manifest=Path("configs/replica_room0_manifest.txt"),
)
```

Localization mode is intentionally not exposed until the narrow C++ helper from
the implementation plan lands. Calling `localize(...)` raises a clear
`NotImplementedError` instead of pretending stock examples can activate
localization mode for a frame range.
