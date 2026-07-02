# ORB-SLAM3 Trajectory, Point Cloud, and Map Localization Notes

This note records the exact workflow used in this repo to run ORB-SLAM3 on a real
trajectory, save `KeyFrameTrajectory.txt`, and build a colored 3D point cloud.
It also outlines the next workflow for building a reusable map and localizing a
robot trajectory segment against that map.

## Context

Workspace:

```bash
/Users/charlie/Projects/ORB_SLAM3
```

Local Docker image:

```bash
orb-slam3:local
```

Installed runner command:

```bash
orb-slam3
```

Default container image in the Python package:

```bash
ghcr.io/velythyl/orb_slam3:latest
```

For local testing, override the image:

```bash
ORB_SLAM3_IMAGE=orb-slam3:local
```

## Dataset Used

The repo did not already contain extracted trajectory image data, only ORB-SLAM3
configs, timestamp lists, ground truth files, and association files. I downloaded
the standard TUM RGB-D sequence:

```bash
mkdir -p datasets
curl -L --fail \
  -o datasets/rgbd_dataset_freiburg1_xyz.tgz \
  https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf datasets/rgbd_dataset_freiburg1_xyz.tgz -C datasets
```

Extracted dataset:

```bash
datasets/rgbd_dataset_freiburg1_xyz
```

Sanity checks:

```bash
wc -l \
  datasets/rgbd_dataset_freiburg1_xyz/rgb.txt \
  datasets/rgbd_dataset_freiburg1_xyz/depth.txt \
  datasets/rgbd_dataset_freiburg1_xyz/groundtruth.txt

find datasets/rgbd_dataset_freiburg1_xyz/rgb -type f | wc -l
```

Observed:

```text
801 rgb.txt
801 depth.txt
3003 groundtruth.txt
798 RGB image files
```

## ORB-SLAM3 Example and Config

The local image contains the built ORB-SLAM3 tree at:

```bash
/opt/orbslam3
```

The relevant executable/config for this run:

```bash
mono_tum
/opt/orbslam3/Vocabulary/ORBvoc.txt
/opt/orbslam3/Examples/Monocular/TUM1.yaml
```

Useful inspection commands:

```bash
docker run --rm orb-slam3:local help

docker run --rm orb-slam3:local bash -lc \
  'command -v mono_tum; command -v stereo_euroc; command -v stereo_inertial_euroc; \
   ls -l /opt/orbslam3/Examples/Monocular/TUM1.yaml /opt/orbslam3/Vocabulary/ORBvoc.txt'
```

Observed executable paths:

```text
/opt/orbslam3/Examples/Monocular/mono_tum
/opt/orbslam3/Examples/Stereo/stereo_euroc
/opt/orbslam3/Examples/Stereo-Inertial/stereo_inertial_euroc
```

## Why a No-Viewer Wrapper Was Needed

Running the stock `mono_tum` example headlessly reached sequence processing, but
then failed because `mono_tum.cc` constructs the ORB-SLAM3 system with
`bUseViewer=true`:

```cpp
ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
```

The failure was:

```text
Pangolin X11: Failed to open X display
```

The image did not include `xvfb-run`, so I compiled a temporary no-viewer wrapper
without changing the ORB-SLAM3 image:

```bash
docker run --rm \
  -v /Users/charlie/Projects/ORB_SLAM3:/work \
  -w /work \
  orb-slam3:local \
  bash -lc 'cp /opt/orbslam3/Examples/Monocular/mono_tum.cc /tmp/mono_tum_noviewer.cc && \
    perl -0pi -e "s/System::MONOCULAR,true/System::MONOCULAR,false/" /tmp/mono_tum_noviewer.cc && \
    g++ -std=c++11 -O3 -DCOMPILEDWITHC11 /tmp/mono_tum_noviewer.cc \
      -o /work/datasets/mono_tum_noviewer \
      -I/opt/orbslam3 \
      -I/opt/orbslam3/include \
      -I/opt/orbslam3/include/CameraModels \
      -I/opt/orbslam3/Thirdparty/Sophus \
      -I/usr/include/eigen3 \
      $(pkg-config --cflags opencv4) \
      -L/opt/orbslam3/lib \
      -L/usr/local/lib \
      -lORB_SLAM3 \
      -lpangolin \
      -lOpenGL \
      -lGLEW \
      -Wl,-rpath,/opt/orbslam3/lib \
      -Wl,-rpath,/usr/local/lib \
      $(pkg-config --libs opencv4) \
      -lpthread'
```

Output:

```bash
datasets/mono_tum_noviewer
```

## Running ORB-SLAM3 and Writing KeyFrameTrajectory.txt

Direct Docker command:

```bash
docker run --rm \
  -v /Users/charlie/Projects/ORB_SLAM3:/work \
  -w /work \
  -e LD_LIBRARY_PATH=/opt/orbslam3/lib:/opt/orbslam3/Thirdparty/DBoW2/lib:/opt/orbslam3/Thirdparty/g2o/lib:/usr/local/lib \
  orb-slam3:local \
  /work/datasets/mono_tum_noviewer \
  /opt/orbslam3/Vocabulary/ORBvoc.txt \
  /opt/orbslam3/Examples/Monocular/TUM1.yaml \
  /work/datasets/rgbd_dataset_freiburg1_xyz
```

Equivalent uv runner command:

```bash
uv run --no-project \
  --with ./dist/orb_slam3-0.1.0-py3-none-any.whl \
  orb-slam3 run \
  --image orb-slam3:local \
  --no-display \
  --env LD_LIBRARY_PATH=/opt/orbslam3/lib:/opt/orbslam3/Thirdparty/DBoW2/lib:/opt/orbslam3/Thirdparty/g2o/lib:/usr/local/lib \
  /work/datasets/mono_tum_noviewer \
  /opt/orbslam3/Vocabulary/ORBvoc.txt \
  /opt/orbslam3/Examples/Monocular/TUM1.yaml \
  /work/datasets/rgbd_dataset_freiburg1_xyz
```

Observed run summary:

```text
Images in the sequence: 798
First KF:0; Map init KF:0
New Map created with 424 points
Shutdown
median tracking time: about 0.011 s
mean tracking time: about 0.012 s
Saving keyframe trajectory to KeyFrameTrajectory.txt ...
```

Output:

```bash
KeyFrameTrajectory.txt
```

The file is in TUM pose format:

```text
timestamp tx ty tz qx qy qz qw
```

ORB-SLAM3 writes keyframe poses as camera-to-world transforms, `Twc`. This is
visible in `src/System.cc` in `System::SaveKeyFrameTrajectoryTUM()`, where it
uses `pKF->GetPoseInverse()`.

## Building the Colored Point Cloud

The host Python did not have `numpy`, `Pillow`, or `cv2`, and the Docker image
has C++ OpenCV but no Python. I added a small C++ utility:

```bash
tools/rgbd_keyframes_to_ply.cpp
```

Compile it inside the ORB-SLAM3 Docker image:

```bash
docker run --rm \
  -v /Users/charlie/Projects/ORB_SLAM3:/work \
  -w /work \
  orb-slam3:local \
  bash -lc 'g++ -std=c++17 -O3 tools/rgbd_keyframes_to_ply.cpp \
    -o datasets/rgbd_keyframes_to_ply \
    $(pkg-config --cflags --libs opencv4)'
```

Generate the PLY:

```bash
mkdir -p pointclouds

docker run --rm \
  -v /Users/charlie/Projects/ORB_SLAM3:/work \
  -w /work \
  orb-slam3:local \
  /work/datasets/rgbd_keyframes_to_ply \
  /work/datasets/rgbd_dataset_freiburg1_xyz \
  /work/KeyFrameTrajectory.txt \
  /work/pointclouds/freiburg1_xyz_orb_keyframes.ply
```

Output:

```bash
pointclouds/freiburg1_xyz_orb_keyframes.ply
```

Observed:

```text
wrote 755396 points to /work/pointclouds/freiburg1_xyz_orb_keyframes.ply
```

Validation:

```bash
python3 -c 'from pathlib import Path
p=Path("pointclouds/freiburg1_xyz_orb_keyframes.ply")
b=p.read_bytes()
h=b.index(b"end_header\n")+len(b"end_header\n")
header=b[:h].decode()
count=int([line.split()[-1] for line in header.splitlines() if line.startswith("element vertex")][0])
print(header, end="")
print("payload_bytes", len(b)-h)
print("expected_bytes", count*15)
print("file_bytes", len(b))'
```

Observed:

```text
ply
format binary_little_endian 1.0
element vertex 755396
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
payload_bytes 11330940
expected_bytes 11330940
file_bytes 11331120
```

Sampled bounds:

```text
sampled_bounds_min [-2.994, -1.807, 0.582]
sampled_bounds_max [1.621, 1.069, 3.624]
```

## Point Cloud Method

The converter does this for each keyframe pose:

1. Read the keyframe timestamp and `Twc` pose from `KeyFrameTrajectory.txt`.
2. Find the nearest RGB frame and nearest depth frame in `rgb.txt` and
   `depth.txt`.
3. Load the RGB and depth PNGs.
4. Backproject depth pixels using TUM1 intrinsics:

```text
fx = 517.306408
fy = 516.469215
cx = 318.643040
cy = 255.313989
depth_scale = 5000
```

5. Transform camera-frame points into the ORB-SLAM3 world frame using `Twc`.
6. Write colored vertices as binary little-endian PLY.

Important caveat: this point cloud uses metric TUM depth images but monocular
ORB-SLAM3 keyframe poses. Monocular SLAM scale is not inherently observable, so
this is best treated as a useful reconstruction/visualization artifact. For a
robot-localization map with reliable metric scale, prefer RGB-D, stereo, or
visual-inertial ORB-SLAM3.

## How to Localize the Robot With Respect to a Map

There are two related goals:

1. Build a persistent ORB-SLAM3 atlas/map from a mapping trajectory.
2. Later load that atlas and track a new robot trajectory segment in
   localization mode, so each returned pose is expressed in the loaded map frame.

### ORB-SLAM3 Hooks

ORB-SLAM3 already has the needed API and YAML settings:

```cpp
System::SaveAtlas(int type)
System::LoadAtlas(int type)
System::ActivateLocalizationMode()
System::DeactivateLocalizationMode()
System::TrackMonocular(...)
```

YAML fields:

```yaml
System.LoadAtlasFromFile: "map_name_without_extension"
System.SaveAtlasToFile: "map_name_without_extension"
```

The implementation appends `.osa` and reads/writes relative to the process
working directory. In this Docker workflow, that means files appear under
`/work`, which is the repo root mounted from the host.

Example:

```yaml
System.SaveAtlasToFile: "maps/freiburg1_xyz_map"
```

will write:

```bash
maps/freiburg1_xyz_map.osa
```

### Recommended Map/Localization Setup

For robust robot localization, use a sensor mode with metric scale:

```text
Best: stereo-inertial or RGB-D
Good: stereo
Works but scale-ambiguous: monocular
```

In this image, the built examples known to exist are:

```text
mono_tum
stereo_euroc
stereo_inertial_euroc
```

For a true metric TUM RGB-D map, the image should also build `rgbd_tum`, or we
should compile a no-viewer RGB-D wrapper just like `mono_tum_noviewer`.

### Concrete Workflow

Create two YAML files from the base camera config:

```bash
mkdir -p maps configs runs
cp Examples/Monocular/TUM1.yaml configs/TUM1_save_map.yaml
cp Examples/Monocular/TUM1.yaml configs/TUM1_load_map_localize.yaml
```

Add this to `configs/TUM1_save_map.yaml`:

```yaml
System.SaveAtlasToFile: "maps/freiburg1_xyz_map"
```

Add this to `configs/TUM1_load_map_localize.yaml`:

```yaml
System.LoadAtlasFromFile: "maps/freiburg1_xyz_map"
```

Then run a mapping pass:

```bash
uv run --no-project \
  --with ./dist/orb_slam3-0.1.0-py3-none-any.whl \
  orb-slam3 run \
  --image orb-slam3:local \
  --no-display \
  --env LD_LIBRARY_PATH=/opt/orbslam3/lib:/opt/orbslam3/Thirdparty/DBoW2/lib:/opt/orbslam3/Thirdparty/g2o/lib:/usr/local/lib \
  /work/datasets/mono_tum_noviewer \
  /opt/orbslam3/Vocabulary/ORBvoc.txt \
  /work/configs/TUM1_save_map.yaml \
  /work/datasets/rgbd_dataset_freiburg1_xyz
```

Expected map output:

```bash
maps/freiburg1_xyz_map.osa
```

For localization, use a separate executable/wrapper that:

1. Constructs `ORB_SLAM3::System` with `System.LoadAtlasFromFile` set in the
   YAML.
2. Calls `SLAM.ActivateLocalizationMode()` before feeding the query trajectory.
3. Feeds only a selected query subsection of frames.
4. Records each returned `Sophus::SE3f` from `TrackMonocular`, `TrackStereo`, or
   `TrackRGBD`.
5. Writes a `CameraTrajectory_TUM.txt` style file.

The important behavioral difference is that localization mode stops local
mapping and tracks against the loaded atlas. Returned poses are then in the map
coordinate frame of the loaded atlas.

Pseudo-code:

```cpp
ORB_SLAM3::System slam(vocab, settings_with_load_atlas,
                       ORB_SLAM3::System::MONOCULAR,
                       false);

slam.ActivateLocalizationMode();

for (Frame frame : query_subsequence) {
    Sophus::SE3f Tcw = slam.TrackMonocular(frame.image, frame.timestamp);
    if (!Tcw.matrix().hasNaN()) {
        Sophus::SE3f Twc = Tcw.inverse();
        save_tum_pose(frame.timestamp, Twc);
    }
}

slam.Shutdown();
```

### Sampling a Subsection of the Robot Trajectory

For a dataset-backed experiment, the cleanest split is:

```text
Mapping segment: frames 0..N
Localization query segment: frames M..K
```

Choose `M` so the query begins in an area with visual overlap with the map.
Pure place recognition can recover from some offset, but localization is much
more reliable if the first query frames are near mapped content.

For TUM-style datasets, make a filtered copy of `rgb.txt` and `depth.txt` for
the query segment, or write the localization wrapper to accept start/end frame
indices and skip frames outside that range.

Example wrapper options worth adding:

```text
--start-frame 300
--end-frame 500
--trajectory-out runs/query_300_500_localized_tum.txt
--activate-localization
```

### What to Build Next

The current work proves:

```text
Docker-backed ORB-SLAM3 runs
KeyFrameTrajectory.txt is produced
RGB-D + keyframe trajectory can produce a colored PLY
```

The next useful engineering step is to add a dedicated no-viewer dataset runner
for map reuse/localization. I would make it support:

```text
mono_tum_noviewer_map
rgbd_tum_noviewer_map, preferred for metric maps
--save-atlas NAME
--load-atlas NAME
--localization-only
--start-frame N
--end-frame N
--save-frame-trajectory FILE
--save-keyframe-trajectory FILE
```

For a robot, the same idea applies online:

1. Run a mapping session and save `my_robot_map.osa`.
2. Start the robot later with `System.LoadAtlasFromFile: "my_robot_map"`.
3. Call `ActivateLocalizationMode()`.
4. Feed camera frames.
5. Use each returned camera/body pose as the robot pose in the map frame,
   applying camera-to-robot extrinsics if the camera frame is not the robot base
   frame.

Frame conversion for robot base pose:

```text
T_map_base = T_map_camera * T_camera_base
```

or equivalently, depending on your calibration convention:

```text
T_map_base = T_map_camera * inverse(T_base_camera)
```

The camera-to-base extrinsic calibration is required if the robot pose should be
reported at `base_link` rather than at the camera optical frame.
