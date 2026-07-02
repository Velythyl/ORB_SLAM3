from __future__ import annotations

from pathlib import Path

import pytest

from pyorbslam3.artifacts import (
    FeatureObservation,
    MapPoint,
    Pose,
    PosedObservation,
    read_map_points_csv,
    read_observations_jsonl,
    read_tum_trajectory,
    write_tum_trajectory,
)


def test_read_tum_trajectory_ignores_blank_and_comment_lines(tmp_path: Path) -> None:
    trajectory = tmp_path / "KeyFrameTrajectory.txt"
    trajectory.write_text(
        "\n"
        "# timestamp tx ty tz qx qy qz qw\n"
        "1.0 2 3 4 0 0 0 1\n",
        encoding="utf-8",
    )

    assert read_tum_trajectory(trajectory) == (Pose(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0),)


def test_read_tum_trajectory_reports_line_number_for_bad_rows(tmp_path: Path) -> None:
    trajectory = tmp_path / "CameraTrajectory.txt"
    trajectory.write_text("1.0 2 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"CameraTrajectory.txt:1: expected 8 TUM fields"):
        read_tum_trajectory(trajectory)


def test_write_tum_trajectory_round_trips(tmp_path: Path) -> None:
    trajectory = tmp_path / "out" / "trajectory.txt"
    poses = (Pose(1.25, 1, 2, 3, 0, 0, 0, 1),)

    write_tum_trajectory(trajectory, poses)

    assert read_tum_trajectory(trajectory) == poses


def test_read_map_points_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "map_points.csv"
    csv_path.write_text(
        "id,x,y,z,observations,found,descriptor_hex\n"
        "7,1.25,2.5,3.75,4,5,abcdef\n",
        encoding="utf-8",
    )

    assert read_map_points_csv(csv_path) == (MapPoint(7, 1.25, 2.5, 3.75, 4, 5, "abcdef"),)


def test_read_observations_jsonl(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        '{"frame_index":0,"timestamp":1.0,"image":"rgb/1.png","depth":null,'
        '"tracking_state":2,"pose":{"tx":1,"ty":2,"tz":3,"qx":0,"qy":0,"qz":0,"qw":1},'
        '"features":[{"index":0,"x":10,"y":20,"size":31,"angle":-1,"response":0.5,'
        '"octave":2,"class_id":-1,"map_point_id":9}]}\n',
        encoding="utf-8",
    )

    assert read_observations_jsonl(observations_path) == (
        PosedObservation(
            frame_index=0,
            timestamp=1.0,
            image="rgb/1.png",
            depth=None,
            tracking_state=2,
            pose=Pose(1.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            features=(FeatureObservation(0, 10.0, 20.0, 31.0, -1.0, 0.5, 2, -1, 9),),
        ),
    )


def test_read_stereo_observations_jsonl(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        '{"frame_index":0,"timestamp":1.0,"image":"left/1.png","right":"right/1.png","depth":null,'
        '"tracking_state":2,"pose":null,"features":[]}\n',
        encoding="utf-8",
    )

    assert read_observations_jsonl(observations_path) == (
        PosedObservation(
            frame_index=0,
            timestamp=1.0,
            image="left/1.png",
            depth=None,
            tracking_state=2,
            pose=None,
            features=(),
            right="right/1.png",
        ),
    )
