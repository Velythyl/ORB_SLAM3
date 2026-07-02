from __future__ import annotations

from pathlib import Path

import pytest

from pyorbslam3 import ImageFrame, RGBDFrame, MonocularRunner, RGBDRunner, StereoFrame, StereoRunner


def test_absolute_path_under_repo_maps_to_work(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    dataset = root / "datasets" / "sequence"
    dataset.mkdir(parents=True)
    runner = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml", repo_root=root)

    assert runner._container_path(dataset) == "/work/datasets/sequence"


def test_missing_absolute_path_is_treated_as_container_path(tmp_path: Path) -> None:
    runner = MonocularRunner(settings=Path("/opt/orbslam3/Examples/Monocular/TUM1.yaml"), repo_root=tmp_path)

    assert runner._container_path(runner.settings) == "/opt/orbslam3/Examples/Monocular/TUM1.yaml"


def test_atlas_suffix_is_normalized(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    runner = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml", repo_root=root)

    yaml_value, final_path = runner._atlas_paths(Path("maps/test_map.osa"))

    assert yaml_value == "/work/maps/test_map"
    assert final_path == root / "maps/test_map.osa"
    assert (root / "maps").exists()


def test_monocular_observation_export_command(tmp_path: Path) -> None:
    runner = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml", repo_root=tmp_path)

    command = runner._observation_command(
        executable=Path("runs/out/helpers/sequence_observation_export"),
        dataset=Path("datasets/sequence"),
        settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
        output_dir=Path("runs/out"),
    )

    assert command == [
        "/work/runs/out/helpers/sequence_observation_export",
        "monocular",
        "/opt/orbslam3/Vocabulary/ORBvoc.txt",
        "/opt/orbslam3/Examples/Monocular/TUM1.yaml",
        "/work/datasets/sequence",
        "/work/runs/out/observations.jsonl",
        "/work/runs/out/map_points.csv",
    ]


def test_monocular_observation_export_command_accepts_manifest(tmp_path: Path) -> None:
    runner = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml", repo_root=tmp_path)

    command = runner._observation_command(
        executable=Path("runs/out/helpers/sequence_observation_export"),
        dataset=Path("datasets/sequence"),
        settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml",
        output_dir=Path("runs/out"),
        manifest=Path("runs/out/configs/sequence_manifest.txt"),
    )

    assert command[-2:] == ["--manifest", "/work/runs/out/configs/sequence_manifest.txt"]


def test_sequence_manifest_is_written_from_frames(tmp_path: Path) -> None:
    runner = MonocularRunner(settings="/opt/orbslam3/Examples/Monocular/TUM1.yaml", repo_root=tmp_path)

    manifest = runner._sequence_manifest(
        output_dir=tmp_path / "runs/out",
        frames=[
            ImageFrame(1.0, "images/000001.png"),
            ImageFrame(1.5, Path("images/000002.png")),
        ],
        manifest=None,
    )

    assert manifest.read_text(encoding="utf-8") == (
        "1.000000000 images/000001.png\n"
        "1.500000000 images/000002.png\n"
    )


def test_rgbd_sequence_manifest_requires_depth(tmp_path: Path) -> None:
    runner = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml", repo_root=tmp_path)

    with pytest.raises(ValueError, match="missing a depth"):
        runner._sequence_manifest(
            output_dir=tmp_path / "runs/out",
            frames=[ImageFrame(1.0, "rgb/000001.png")],
            manifest=None,
        )


def test_rgbd_sequence_manifest_accepts_explicit_rgbd_frames(tmp_path: Path) -> None:
    runner = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml", repo_root=tmp_path)

    manifest = runner._sequence_manifest(
        output_dir=tmp_path / "runs/out",
        frames=[RGBDFrame(1.0, "rgb/000001.png", "depth/000001.png")],
        manifest=None,
    )

    assert manifest.read_text(encoding="utf-8") == "1.000000000 rgb/000001.png depth/000001.png\n"


def test_stereo_sequence_manifest_is_written_from_frames(tmp_path: Path) -> None:
    runner = StereoRunner(settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml", repo_root=tmp_path)

    manifest = runner._sequence_manifest(
        output_dir=tmp_path / "runs/out",
        frames=[
            StereoFrame(1.0, "mav0/cam0/data/000001.png", "mav0/cam1/data/000001.png"),
            StereoFrame(1.5, Path("left/000002.png"), Path("right/000002.png")),
        ],
        manifest=None,
    )

    assert manifest.read_text(encoding="utf-8") == (
        "1.000000000 mav0/cam0/data/000001.png mav0/cam1/data/000001.png\n"
        "1.500000000 left/000002.png right/000002.png\n"
    )


def test_stereo_sequence_manifest_rejects_image_frame(tmp_path: Path) -> None:
    runner = StereoRunner(settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml", repo_root=tmp_path)

    with pytest.raises(ValueError, match="Stereo frame 0"):
        runner._sequence_manifest(
            output_dir=tmp_path / "runs/out",
            frames=[ImageFrame(1.0, "left/000001.png")],
            manifest=None,
        )


def test_stereo_observation_requires_manifest(tmp_path: Path) -> None:
    runner = StereoRunner(settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml", repo_root=tmp_path)

    with pytest.raises(ValueError, match="Stereo runs require"):
        runner._observation_command(
            executable=Path("runs/out/helpers/sequence_observation_export"),
            dataset=Path("datasets/sequence"),
            settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml",
            output_dir=Path("runs/out"),
        )


def test_stereo_observation_accepts_manifest(tmp_path: Path) -> None:
    runner = StereoRunner(settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml", repo_root=tmp_path)

    command = runner._observation_command(
        executable=Path("runs/out/helpers/sequence_observation_export"),
        dataset=Path("datasets/sequence"),
        settings="/opt/orbslam3/Examples/Stereo/EuRoC.yaml",
        output_dir=Path("runs/out"),
        manifest=Path("custom_stereo.txt"),
    )

    assert command[:7] == [
        "/work/runs/out/helpers/sequence_observation_export",
        "stereo",
        "/opt/orbslam3/Vocabulary/ORBvoc.txt",
        "/opt/orbslam3/Examples/Stereo/EuRoC.yaml",
        "/work/datasets/sequence",
        "/work/runs/out/observations.jsonl",
        "/work/runs/out/map_points.csv",
    ]
    assert command[-2:] == ["--manifest", "/work/custom_stereo.txt"]


def test_rgbd_observation_requires_manifest(tmp_path: Path) -> None:
    runner = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml", repo_root=tmp_path)

    with pytest.raises(ValueError, match="RGB-D runs require"):
        runner._observation_command(
            executable=Path("runs/out/helpers/sequence_observation_export"),
            dataset=Path("datasets/sequence"),
            settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml",
            output_dir=Path("runs/out"),
        )


def test_rgbd_observation_accepts_manifest(tmp_path: Path) -> None:
    runner = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml", repo_root=tmp_path)

    command = runner._observation_command(
        executable=Path("runs/out/helpers/sequence_observation_export"),
        dataset=Path("datasets/sequence"),
        settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml",
        output_dir=Path("runs/out"),
        manifest=Path("custom_sequence.txt"),
    )

    assert command[-2:] == ["--manifest", "/work/custom_sequence.txt"]


def test_rgbd_manifest_run_uses_dataset_agnostic_helper_without_association(tmp_path: Path, monkeypatch) -> None:
    runner = RGBDRunner(settings="/opt/orbslam3/Examples/RGB-D/TUM1.yaml", repo_root=tmp_path)
    captured = {}

    def fake_ensure_helper(output_dir: Path) -> Path:
        return output_dir / "helpers" / "sequence_observation_export"

    def fake_execute_run(**kwargs):
        captured.update(kwargs)
        from pyorbslam3 import RunResult

        return RunResult(
            output_dir=kwargs["output_dir"],
            keyframe_trajectory=None,
            camera_trajectory=None,
            atlas=kwargs["atlas"],
            observations_path=None,
            map_points_path=None,
            log=kwargs["log"],
            poses=(),
            runner=runner,
            dataset=Path("datasets/replica/room0"),
            sensor="rgbd",
            sequence_manifest=kwargs["sequence_manifest"],
        )

    monkeypatch.setattr(runner, "_ensure_observation_helper", fake_ensure_helper)
    monkeypatch.setattr(runner, "_execute_run", fake_execute_run)

    result = runner.run_dataset(
        dataset=Path("datasets/replica/room0"),
        output_dir=tmp_path / "runs/replica_room0",
        frames=[ImageFrame(0.0, "rgb/000000.png", "depth/000000.png")],
    )

    assert result.sequence_manifest == tmp_path / "runs/replica_room0/configs/sequence_manifest.txt"
    assert captured["command"][-2:] == ["--manifest", "/work/runs/replica_room0/configs/sequence_manifest.txt"]
    assert "associations.txt" not in captured["command"]
