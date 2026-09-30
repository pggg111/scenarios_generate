#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from PIL import Image


DEFAULT_BEHAVIOR_TREE = "BTRegularNav.xml"
DEFAULT_MODEL = "pedestrian"
DEFAULT_VELOCITY = 0.8
DEFAULT_DESIRED_VELOCITY = 1.0


class ScenarioYamlDumper(yaml.SafeDumper):
    pass


def represent_list(dumper: yaml.Dumper, data: list[Any]) -> yaml.SequenceNode:
    flow_style = bool(data) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)


ScenarioYamlDumper.add_representer(list, represent_list)


@dataclass(frozen=True)
class CameraRayMapping:
    image_width: int
    image_height: int
    camera_projection: np.ndarray
    camera_world_transform: np.ndarray
    ground_plane_z: float

    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        ndc_x = 2.0 * ((pixel_x + 0.5) / self.image_width) - 1.0
        ndc_y = 1.0 - 2.0 * ((pixel_y + 0.5) / self.image_height)

        clip_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
        clip_far = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)

        inv_projection = np.linalg.inv(self.camera_projection)
        near_camera = inv_projection @ clip_near
        far_camera = inv_projection @ clip_far
        near_camera /= near_camera[3]
        far_camera /= far_camera[3]

        near_world = near_camera @ self.camera_world_transform
        far_world = far_camera @ self.camera_world_transform
        near_world /= near_world[3]
        far_world /= far_world[3]

        ray_direction = far_world[:3] - near_world[:3]
        if abs(ray_direction[2]) < 1e-9:
            raise ValueError("Camera ray is parallel to the ground plane")

        t = (self.ground_plane_z - near_world[2]) / ray_direction[2]
        hit_world = near_world[:3] + t * ray_direction
        return float(hit_world[0]), float(hit_world[1])


@dataclass
class Point:
    pixel: tuple[float, float]
    waypoint_id: str | None = None


@dataclass
class Track:
    kind: str
    name: str
    color: str
    points: list[Point] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually click robot/human trajectories on a GRScenes top-down image and export scenario.yaml."
    )
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing topdown_mapping.json and images.")
    parser.add_argument(
        "--output",
        help="Output scenario YAML. Defaults to <scene-dir>/manual_scenario.yaml.",
    )
    parser.add_argument(
        "--background",
        choices=("overlay", "topdown", "navmesh"),
        default="overlay",
        help="Raw image to click on. Default: overlay.png.",
    )
    parser.add_argument("--mapping", help="Defaults to <scene-dir>/topdown_mapping.json.")
    parser.add_argument("--waypoints", help="Defaults to <scene-dir>/waypoints_by_region.json.")
    parser.add_argument(
        "--snap-radius",
        type=float,
        default=18.0,
        help="Snap clicks to the nearest waypoint within this pixel radius. Use 0 to disable.",
    )
    parser.add_argument("--hide-waypoints", action="store_true", help="Do not draw waypoint reference dots.")
    parser.add_argument("--show-waypoint-names", action="store_true", help="Draw waypoint ids next to reference dots.")
    parser.add_argument("--no-snap", action="store_true", help="Disable waypoint snapping.")
    parser.add_argument("--round", type=int, default=5, help="Decimal places for world coordinates/yaw.")
    parser.add_argument("--robot-behavior", default="manually clicked robot route")
    parser.add_argument("--human-velocity", type=float, default=DEFAULT_VELOCITY)
    parser.add_argument("--human-desired-velocity", type=float, default=DEFAULT_DESIRED_VELOCITY)
    parser.add_argument("--no-show", action="store_true", help="Write an empty/default preview without opening UI.")
    return parser.parse_args()


def load_mapping(mapping_path: Path) -> CameraRayMapping:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    camera_geometry = data.get("camera_geometry")
    if not camera_geometry or camera_geometry.get("method") != "camera_ray_ground_intersection":
        raise ValueError(f"{mapping_path} does not contain camera ray ground mapping")

    return CameraRayMapping(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        camera_projection=np.array(camera_geometry["camera_projection"], dtype=np.float64).reshape((4, 4), order="F"),
        camera_world_transform=np.array(camera_geometry["camera_world_transform"], dtype=np.float64).reshape((4, 4)),
        ground_plane_z=float(camera_geometry["ground_plane_z"]),
    )


def background_path(scene_dir: Path, background: str) -> Path:
    candidates = {
        "overlay": [scene_dir / "overlay.png", scene_dir / "topdown.png", scene_dir / "navmesh_mask.png"],
        "topdown": [scene_dir / "topdown.png", scene_dir / "overlay.png", scene_dir / "navmesh_mask.png"],
        "navmesh": [scene_dir / "navmesh_mask.png", scene_dir / "overlay.png", scene_dir / "topdown.png"],
    }
    for path in candidates[background]:
        if path.exists():
            return path
    raise FileNotFoundError(f"No usable background image found in {scene_dir}")


def load_waypoints(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    waypoints: list[dict[str, Any]] = []
    for region in data.get("waypoints_by_region", []):
        for waypoint in region.get("waypoints", []):
            pixel = waypoint.get("pixel")
            if isinstance(pixel, list) and len(pixel) >= 2:
                waypoints.append(
                    {
                        "id": str(waypoint.get("id") or ""),
                        "pixel": (float(pixel[0]), float(pixel[1])),
                    }
                )
    return waypoints


def nearest_waypoint(
    pixel: tuple[float, float],
    waypoints: list[dict[str, Any]],
    snap_radius: float,
) -> tuple[tuple[float, float], str | None]:
    if snap_radius <= 0 or not waypoints:
        return pixel, None
    px, py = pixel
    nearest: dict[str, Any] | None = None
    nearest_dist = float("inf")
    for waypoint in waypoints:
        wx, wy = waypoint["pixel"]
        dist = math.hypot(wx - px, wy - py)
        if dist < nearest_dist:
            nearest = waypoint
            nearest_dist = dist
    if nearest is not None and nearest_dist <= snap_radius:
        return nearest["pixel"], nearest["id"]
    return pixel, None


def yaw_degrees(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def rounded(value: float, places: int) -> float:
    return round(float(value), places)


def route_to_pose_and_waypoints(
    track: Track,
    mapping: CameraRayMapping,
    places: int,
) -> tuple[list[float], list[list[float]]]:
    if len(track.points) < 2:
        raise ValueError(f"{track.name} needs at least 2 clicked points")

    world_points = [mapping.pixel_to_world(*point.pixel) for point in track.points]
    route_yaws = [
        yaw_degrees(world_points[index], world_points[index + 1])
        for index in range(len(world_points) - 1)
    ]
    last_yaw = route_yaws[-1] if route_yaws else 0.0
    waypoint_yaws = [*route_yaws[1:], last_yaw]

    pose = [
        rounded(world_points[0][0], places),
        rounded(world_points[0][1], places),
        rounded(route_yaws[0], places),
    ]
    waypoints = [
        [
            rounded(world_xy[0], places),
            rounded(world_xy[1], places),
            rounded(waypoint_yaws[index], places),
        ]
        for index, world_xy in enumerate(world_points[1:])
    ]
    return pose, waypoints


def scenario_from_tracks(tracks: list[Track], mapping: CameraRayMapping, args: argparse.Namespace) -> dict[str, Any]:
    scenario: dict[str, Any] = {"dynamic": []}
    human_index = 0

    for track in tracks:
        if len(track.points) < 2:
            continue
        pose, waypoints = route_to_pose_and_waypoints(track, mapping, args.round)
        if track.kind == "robot":
            scenario["robot"] = {
                "name": "robot",
                "pose": pose,
                "waypoints": waypoints,
                "behavior": args.robot_behavior,
            }
        else:
            human_index += 1
            scenario["dynamic"].append(
                {
                    "name": f"hunav_{human_index}",
                    "model": DEFAULT_MODEL,
                    "pose": pose,
                    "behavior_tree": DEFAULT_BEHAVIOR_TREE,
                    "velocity": rounded(args.human_velocity, args.round),
                    "desired_velocity": rounded(args.human_desired_velocity, args.round),
                    "waypoints": waypoints,
                }
            )

    return scenario


def write_scenario(path: Path, scenario: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Manually clicked topdown scenario\n\n" + yaml.dump(
        scenario,
        Dumper=ScenarioYamlDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def draw_waypoint_reference(
    ax: plt.Axes,
    waypoints: list[dict[str, Any]],
    show_names: bool,
) -> None:
    if not waypoints:
        return
    xs = [waypoint["pixel"][0] for waypoint in waypoints]
    ys = [waypoint["pixel"][1] for waypoint in waypoints]
    ax.scatter(xs, ys, s=12, color="#2c7fb8", alpha=0.62, edgecolors="white", linewidths=0.25)
    if show_names:
        for waypoint in waypoints:
            pixel_x, pixel_y = waypoint["pixel"]
            ax.text(pixel_x + 3, pixel_y + 3, waypoint["id"], color="#08519c", fontsize=5.5, alpha=0.85)


def draw_tracks(ax: plt.Axes, tracks: list[Track], active: Track | None) -> None:
    for track in tracks:
        if not track.points:
            continue
        xs = [point.pixel[0] for point in track.points]
        ys = [point.pixel[1] for point in track.points]
        line_style = "--" if track.kind == "robot" else "-"
        width = 3.0 if track is active else 2.2
        ax.plot(xs, ys, color=track.color, linestyle=line_style, linewidth=width, alpha=0.95)
        ax.scatter(xs[:1], ys[:1], marker="D" if track.kind == "robot" else "s", s=90, color=track.color, edgecolors="white")
        if len(xs) > 1:
            ax.scatter(xs[1:], ys[1:], marker="o", s=65, color=track.color, edgecolors="white")
        for index, point in enumerate(track.points):
            label = "S" if index == 0 else str(index)
            if point.waypoint_id:
                label = f"{label}:{point.waypoint_id.split('_')[-1]}"
            ax.text(
                point.pixel[0] + 6,
                point.pixel[1] + 6,
                f"{track.name} {label}",
                color=track.color,
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.2},
            )


def save_preview(
    path: Path,
    image: Image.Image,
    tracks: list[Track],
    waypoints: list[dict[str, Any]],
    show_waypoints: bool,
    show_waypoint_names: bool,
) -> None:
    fig_width = 12
    fig_height = max(8, fig_width * image.height / image.width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if image.mode == "L":
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(image)
    if show_waypoints:
        draw_waypoint_reference(ax, waypoints, show_waypoint_names)
    draw_tracks(ax, tracks, None)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.set_title(path.name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else scene_dir / "manual_scenario.yaml"
    mapping_path = Path(args.mapping).expanduser().resolve() if args.mapping else scene_dir / "topdown_mapping.json"
    waypoint_path = Path(args.waypoints).expanduser().resolve() if args.waypoints else scene_dir / "waypoints_by_region.json"
    snap_radius = 0.0 if args.no_snap else args.snap_radius

    mapping = load_mapping(mapping_path)
    image_path = background_path(scene_dir, args.background)
    image = Image.open(image_path).convert("RGB")
    if image.size != (mapping.image_width, mapping.image_height):
        print(
            f"Warning: background size {image.size} differs from mapping size "
            f"{(mapping.image_width, mapping.image_height)}; click conversion still uses mapping.",
            flush=True,
        )
    waypoints = load_waypoints(waypoint_path)
    preview_path = output_path.with_name(f"{output_path.stem}_preview.png")

    tracks: list[Track] = []
    active: Track | None = None
    human_count = 0
    colors = list(plt.get_cmap("tab10").colors)

    def set_robot() -> None:
        nonlocal active
        robot = next((track for track in tracks if track.kind == "robot"), None)
        if robot is None:
            robot = Track("robot", "robot", "black")
            tracks.append(robot)
        active = robot
        print("Active track: robot", flush=True)

    def new_human() -> None:
        nonlocal active, human_count
        human_count += 1
        color = colors[(human_count - 1) % len(colors)]
        track = Track("human", f"human{human_count}", color)
        tracks.append(track)
        active = track
        print(f"Active track: {track.name}", flush=True)

    set_robot()

    def redraw(ax: plt.Axes) -> None:
        ax.clear()
        ax.imshow(image)
        if not args.hide_waypoints:
            draw_waypoint_reference(ax, waypoints, args.show_waypoint_names)
        draw_tracks(ax, tracks, active)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        active_name = active.name if active else "none"
        ax.set_title(
            "Manual scenario editor | r:robot h:new human u:undo s:save q:quit | "
            f"active={active_name} snap={snap_radius:g}px"
        )

    def save_all() -> None:
        scenario = scenario_from_tracks(tracks, mapping, args)
        write_scenario(output_path, scenario)
        save_preview(
            preview_path,
            image,
            tracks,
            waypoints,
            not args.hide_waypoints,
            args.show_waypoint_names,
        )

    if args.no_show:
        save_all()
        return

    fig_width = 12
    fig_height = max(8, fig_width * image.height / image.width)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    redraw(ax)
    fig.tight_layout()

    def on_click(event: MouseEvent) -> None:
        nonlocal active
        if event.button != MouseButton.LEFT or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if active is None:
            new_human()
        assert active is not None
        pixel = (
            max(0.0, min(mapping.image_width - 1.0, float(event.xdata))),
            max(0.0, min(mapping.image_height - 1.0, float(event.ydata))),
        )
        snapped_pixel, waypoint_id = nearest_waypoint(pixel, waypoints, snap_radius)
        active.points.append(Point(snapped_pixel, waypoint_id))
        world = mapping.pixel_to_world(*snapped_pixel)
        snap_text = f" snapped={waypoint_id}" if waypoint_id else ""
        print(
            f"{active.name} point {len(active.points) - 1}: "
            f"pixel=({snapped_pixel[0]:.1f}, {snapped_pixel[1]:.1f}) "
            f"world=({world[0]:.4f}, {world[1]:.4f}){snap_text}",
            flush=True,
        )
        redraw(ax)
        fig.canvas.draw_idle()

    def on_key(event: KeyEvent) -> None:
        if event.key == "r":
            set_robot()
        elif event.key == "h":
            new_human()
        elif event.key == "u":
            if active is not None and active.points:
                removed = active.points.pop()
                print(f"Undo {active.name}: {removed.pixel}", flush=True)
        elif event.key == "s":
            save_all()
        elif event.key == "q":
            plt.close(fig)
            return
        else:
            return
        redraw(ax)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
