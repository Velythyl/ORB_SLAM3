from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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

    def as_tuple(self) -> tuple[float, float, float, float, float, float, float, float]:
        return (self.timestamp, self.tx, self.ty, self.tz, self.qx, self.qy, self.qz, self.qw)

    def as_matrix(self) -> "Any":
        np = _numpy()
        x, y, z, w = self.qx, self.qy, self.qz, self.qw
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
        matrix[:3, 3] = [self.tx, self.ty, self.tz]
        return matrix


@dataclass(frozen=True)
class MapPoint:
    id: int
    x: float
    y: float
    z: float
    observations: int
    found: int
    descriptor_hex: str = ""


@dataclass(frozen=True)
class FeatureObservation:
    index: int
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int
    map_point_id: int | None = None


@dataclass(frozen=True)
class PosedObservation:
    frame_index: int
    timestamp: float
    image: str
    depth: str | None
    tracking_state: int
    pose: Pose | None = None
    features: tuple[FeatureObservation, ...] = ()
    right: str | None = None


@dataclass(frozen=True)
class PointCloud:
    points: "Any"
    colors: "Any | None" = None


def read_tum_trajectory(path: Path) -> tuple[Pose, ...]:
    poses: list[Pose] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM fields "
                    f"(timestamp tx ty tz qx qy qz qw), got {len(fields)}"
                )
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: TUM trajectory row contains a non-float value") from exc
            poses.append(Pose(*values))

    return tuple(poses)


def poses_to_numpy(poses: Iterable[Pose]) -> "Any":
    np = _numpy()
    return np.asarray([pose.as_tuple() for pose in poses], dtype=float)


def poses_to_matrices(poses: Iterable[Pose]) -> "Any":
    np = _numpy()
    return np.stack([pose.as_matrix() for pose in poses])


def write_tum_trajectory(path: Path, poses: Iterable[Pose]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pose in poses:
            handle.write(
                f"{pose.timestamp:.9f} {pose.tx:.9f} {pose.ty:.9f} {pose.tz:.9f} "
                f"{pose.qx:.9f} {pose.qy:.9f} {pose.qz:.9f} {pose.qw:.9f}\n"
            )


def read_ply_vertex_count(path: Path) -> int:
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"{path}: PLY header does not contain an element vertex row")


def read_map_points_csv(path: Path) -> tuple[MapPoint, ...]:
    points: list[MapPoint] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            points.append(
                MapPoint(
                    id=int(row["id"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    observations=int(row["observations"]),
                    found=int(row["found"]),
                    descriptor_hex=row.get("descriptor_hex", ""),
                )
            )
    return tuple(points)


def read_observations_jsonl(path: Path) -> tuple[PosedObservation, ...]:
    observations: list[PosedObservation] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON observation row") from exc
            observations.append(_decode_observation(payload, path=Path(path), line_number=line_number))
    return tuple(observations)


def point_positions_to_numpy(points: Iterable[MapPoint]) -> "Any":
    np = _numpy()
    return np.asarray([(point.x, point.y, point.z) for point in points], dtype=float)


def read_ply_pointcloud(path: Path) -> PointCloud:
    np = _numpy()
    path = Path(path)
    with path.open("rb") as handle:
        vertex_count: int | None = None
        properties: list[tuple[str, str]] = []
        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"{path}: PLY header is incomplete")
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
                continue
            if line.startswith("property "):
                _, kind, name = line.split()
                properties.append((kind, name))
                continue
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{path}: PLY header does not contain an element vertex row")

        dtype_fields: list[tuple[str, str]] = []
        for kind, name in properties:
            if kind == "float":
                dtype_fields.append((name, "<f4"))
            elif kind == "uchar":
                dtype_fields.append((name, "u1"))
            else:
                raise ValueError(f"{path}: unsupported PLY property type {kind!r}")
        data = np.frombuffer(handle.read(), dtype=np.dtype(dtype_fields), count=vertex_count)

    points = np.column_stack([data["x"], data["y"], data["z"]])
    color_names = {"red", "green", "blue"}
    colors = np.column_stack([data["red"], data["green"], data["blue"]]) if color_names.issubset(data.dtype.names or ()) else None
    return PointCloud(points=points, colors=colors)


def to_open3d_pointcloud(pointcloud: PointCloud | str | Path) -> Any:
    o3d = _open3d()
    np = _numpy()
    cloud = read_ply_pointcloud(Path(pointcloud)) if isinstance(pointcloud, (str, Path)) else pointcloud
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(cloud.points, dtype=float))
    if cloud.colors is not None:
        colors = np.asarray(cloud.colors, dtype=float)
        if colors.size and colors.max() > 1.0:
            colors = colors / 255.0
        geometry.colors = o3d.utility.Vector3dVector(colors)
    return geometry


def view_pointcloud(pointcloud: PointCloud | str | Path, *, window_name: str = "ORB-SLAM3 Point Cloud") -> None:
    o3d = _open3d()
    geometry = to_open3d_pointcloud(pointcloud)
    o3d.visualization.draw_geometries([geometry], window_name=window_name)


def _decode_observation(payload: dict[str, Any], *, path: Path, line_number: int) -> PosedObservation:
    pose_payload = payload.get("pose")
    pose = None
    if pose_payload is not None:
        try:
            pose = Pose(
                float(payload["timestamp"]),
                float(pose_payload["tx"]),
                float(pose_payload["ty"]),
                float(pose_payload["tz"]),
                float(pose_payload["qx"]),
                float(pose_payload["qy"]),
                float(pose_payload["qz"]),
                float(pose_payload["qw"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid pose payload") from exc

    features = []
    for feature in payload.get("features", ()):
        try:
            map_point_id = feature.get("map_point_id")
            features.append(
                FeatureObservation(
                    index=int(feature["index"]),
                    x=float(feature["x"]),
                    y=float(feature["y"]),
                    size=float(feature["size"]),
                    angle=float(feature["angle"]),
                    response=float(feature["response"]),
                    octave=int(feature["octave"]),
                    class_id=int(feature["class_id"]),
                    map_point_id=int(map_point_id) if map_point_id is not None else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid feature payload") from exc

    try:
        return PosedObservation(
            frame_index=int(payload["frame_index"]),
            timestamp=float(payload["timestamp"]),
            image=str(payload["image"]),
            depth=str(payload["depth"]) if payload.get("depth") is not None else None,
            tracking_state=int(payload["tracking_state"]),
            pose=pose,
            features=tuple(features),
            right=str(payload["right"]) if payload.get("right") is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid observation payload") from exc


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install numpy to use this convenience conversion.") from exc
    return np


def _open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Install open3d to view point clouds.") from exc
    return o3d
