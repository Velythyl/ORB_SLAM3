# Prompt For Next Agent

You are working in `/Users/charlie/Projects/ORB_SLAM3`.

The user wants to expand the current `pyorbslam3` thin Docker-backed wrapper so
it supports stereo, all visual-inertial ORB-SLAM3 modes, atlas loading, and
map-posed observation export after building a map.

Read this plan first:

```text
plans/stereo_inertial_atlas_observations.md
```

Then inspect the current implementation:

```text
pyorbslam3/api.py
pyorbslam3/artifacts.py
pyorbslam3/docker.py
tools/sequence_observation_export.cpp
include/System.h
docs/thin_python_wrapper.md
```

Current state to preserve:

- `MonocularRunner` and `RGBDRunner` exist.
- `run_sequence`, `run_dataset`, `build_map`, and `build_map_sequence` work for
  the current monocular/RGB-D helper flow.
- `localize()` currently raises `NotImplementedError`.
- `tools/sequence_observation_export.cpp` currently supports only `monocular`
  and `rgbd`.
- `build_map()` already appends `System.SaveAtlasToFile` through a generated
  settings overlay.

Important design constraints:

- This project intentionally avoids native Python bindings for now.
- Use narrow C++ adapter executable support to expose ORB-SLAM3 C++ APIs to
  Python-friendly artifacts.
- Do not depend on large datasets for the normal unit test suite.
- Preserve existing public behavior where practical.
- Be explicit about pose semantics: runtime tracking observations are not
  always the same as final optimized map/keyframe poses after loop closure.

Recommended first implementation slice:

1. Add explicit dataclasses for `StereoFrame`, `RGBDFrame`, and `IMUPoint` while
   keeping current `ImageFrame` compatibility.
2. Refactor manifest generation in `pyorbslam3/api.py` around sensor-specific
   frame formats.
3. Add `StereoRunner`.
4. Extend or replace `tools/sequence_observation_export.cpp` so it accepts a
   stereo manifest and calls `TrackStereo`.
5. Add parser/test coverage for the new manifest and artifact shapes.

After stereo works, continue with shared IMU plumbing, inertial runners, atlas
loading/localization mode, and final-map/keyframe observation export according
to the plan.

When implementing:

- Use `apply_patch` for manual edits.
- Prefer `rg` and `sed` for inspection.
- Do not revert unrelated existing changes; the worktree may already be dirty.
- Run focused tests if available. If Docker integration is needed and fails due
  to sandbox/network restrictions, request escalation with a clear justification.
