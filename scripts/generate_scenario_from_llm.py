#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml


DEFAULT_BEHAVIOR_TREE = "BTRegularNav.xml"
DEFAULT_MODEL = "pedestrian"
DEFAULT_VELOCITY = 0.8
DEFAULT_DESIRED_VELOCITY = 1.0
AVAILABLE_AGENT_MODELS = (
    "pedestrian",
    "doctor",
    "police",
    "construction_worker",
)
AGENT_MODEL_LIMITS = {
    "doctor": 2,
    "police": 4,
    "construction_worker": 4,
}
TOKEN_USAGE_PREFIX = "TOKEN_USAGE "


class ScenarioYamlDumper(yaml.SafeDumper):
    pass


def represent_list(dumper: yaml.Dumper, data: list[Any]) -> yaml.SequenceNode:
    flow_style = bool(data) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)


ScenarioYamlDumper.add_representer(list, represent_list)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    reported: bool = False

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "reported": self.reported,
        }


@dataclass(frozen=True)
class LLMCallResult:
    data: Any
    usage: TokenUsage


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the LLM to select pedestrian spawn/waypoint pixels on overlay.png, "
            "then convert them into a HuNav scenario YAML."
        )
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing overlay.png, topdown.json, and topdown_mapping.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated scenario YAML.",
    )
    parser.add_argument(
        "--mapping",
        help="Optional explicit topdown_mapping.json path. Defaults to <scene-dir>/topdown_mapping.json.",
    )
    parser.add_argument(
        "--llm-config",
        required=True,
        help="YAML config containing Gemini-compatible API settings.",
    )
    parser.add_argument(
        "--prompt",
        help="Extra user request for the LLM, for example desired number of pedestrians and behavior.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a text file containing the extra user request for the LLM.",
    )
    parser.add_argument(
        "--scene-prompt-file",
        help=(
            "Path to scene-specific rules. Defaults to <scene-dir>/scene_prompt.txt "
            "when the file exists."
        ),
    )
    parser.add_argument(
        "--save-llm-output",
        help="Optional path to save the raw parsed LLM JSON response before world-coordinate conversion.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Build and print the LLM prompt without sending an API request.",
    )
    parser.add_argument(
        "--robot-intermediate-waypoints",
        action="store_true",
        help=(
            "Temporary test mode: ask the LLM to output robot intermediate waypoints "
            "and keep them in the generated YAML."
        ),
    )
    parser.add_argument(
        "--round",
        type=int,
        default=5,
        help="Decimal places for generated world coordinates and yaw values.",
    )
    parser.add_argument(
        "--default-model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_AGENT_MODELS,
        help=(
            "Occupation category used when an agent does not specify model. "
            f"Default: {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--default-velocity",
        type=float,
        default=DEFAULT_VELOCITY,
        help=f"Velocity used when an agent does not specify one. Default: {DEFAULT_VELOCITY}.",
    )
    parser.add_argument(
        "--default-desired-velocity",
        type=float,
        default=DEFAULT_DESIRED_VELOCITY,
        help=(
            "Desired velocity used when an agent does not specify one. "
            f"Default: {DEFAULT_DESIRED_VELOCITY}."
        ),
    )
    parser.add_argument(
        "--default-behavior-tree",
        default=DEFAULT_BEHAVIOR_TREE,
        help=f"Behavior tree used when an agent does not specify one. Default: {DEFAULT_BEHAVIOR_TREE}.",
    )
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def config_value(config: dict[str, Any], key: str, *, required: bool = True) -> str:
    value = config.get(key)
    env_key = config.get(f"{key}_env")
    if not value and env_key:
        value = os.environ.get(str(env_key))
    if required and not value:
        if env_key:
            raise ValueError(f"LLM config is missing {key}; env {env_key} is not set")
        raise ValueError(f"LLM config is missing {key}")
    return str(value or "")


def load_mapping(mapping_path: Path) -> CameraRayMapping:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    camera_geometry = data.get("camera_geometry")
    if not camera_geometry or camera_geometry.get("method") != "camera_ray_ground_intersection":
        raise ValueError(
            f"{mapping_path} does not contain camera_geometry.method=camera_ray_ground_intersection"
        )

    return CameraRayMapping(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        camera_projection=np.array(camera_geometry["camera_projection"], dtype=np.float64).reshape(
            (4, 4), order="F"
        ),
        camera_world_transform=np.array(
            camera_geometry["camera_world_transform"], dtype=np.float64
        ).reshape((4, 4)),
        ground_plane_z=float(camera_geometry["ground_plane_z"]),
    )


def agents_from_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        agents = data
    elif isinstance(data, dict) and isinstance(data.get("agents"), list):
        agents = data["agents"]
    else:
        raise ValueError("LLM output must be either an agents array or an object with an agents array")

    if not agents:
        raise ValueError("LLM output contains no agents")
    if not all(isinstance(agent, dict) for agent in agents):
        raise ValueError("Every agent entry must be a JSON object")
    return agents


def load_region_annotations(scene_dir: Path) -> dict[str, Any] | None:
    topdown_json = scene_dir / "topdown.json"
    if not topdown_json.exists():
        return None
    data = json.loads(topdown_json.read_text(encoding="utf-8"))
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return None

    regions: list[dict[str, Any]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        points = shape.get("points")
        if not isinstance(label, str) or not isinstance(points, list):
            continue
        regions.append(
            {
                "id": index,
                "label": label,
                "shape_type": shape.get("shape_type", "polygon"),
                "points": points,
            }
        )

    if not regions:
        return None
    return {"regions": regions}


def load_waypoint_context(scene_dir: Path) -> dict[str, Any]:
    waypoint_path = scene_dir / "waypoints_by_region.json"
    if not waypoint_path.exists():
        raise FileNotFoundError(f"Missing required waypoint list: {waypoint_path}")
    data = json.loads(waypoint_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{waypoint_path} must contain a JSON object")
    if not isinstance(data.get("region_areas"), list) or not isinstance(
        data.get("waypoints_by_region"), list
    ):
        raise ValueError(
            f"{waypoint_path} must contain region_areas and waypoints_by_region arrays"
        )
    return data


def build_waypoint_lookup(waypoint_context: dict[str, Any]) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {}
    for region in waypoint_context.get("waypoints_by_region", []):
        if not isinstance(region, dict):
            continue
        waypoints = region.get("waypoints")
        if not isinstance(waypoints, list):
            continue
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                continue
            waypoint_id = waypoint.get("id")
            pixel = waypoint.get("pixel")
            if not isinstance(waypoint_id, str) or not isinstance(pixel, list) or len(pixel) < 2:
                continue
            lookup[waypoint_id] = (float(pixel[0]), float(pixel[1]))
    if not lookup:
        raise ValueError("waypoints_by_region.json does not contain any waypoint ids")
    return lookup


def build_waypoint_region_lookup(waypoint_context: dict[str, Any]) -> dict[str, tuple[int | None, str]]:
    lookup: dict[str, tuple[int | None, str]] = {}
    for region in waypoint_context.get("waypoints_by_region", []):
        if not isinstance(region, dict):
            continue
        raw_region_id = region.get("region_id")
        region_id = int(raw_region_id) if isinstance(raw_region_id, int) else None
        label = str(region.get("label") or "")
        waypoints = region.get("waypoints")
        if not isinstance(waypoints, list):
            continue
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                continue
            waypoint_id = waypoint.get("id")
            if isinstance(waypoint_id, str):
                lookup[waypoint_id] = (region_id, label)
    return lookup


def transition_waypoint_ids(waypoint_context: dict[str, Any]) -> set[str]:
    waypoint_ids: set[str] = set()
    transitions = waypoint_context.get("region_transitions")
    if not isinstance(transitions, list):
        return waypoint_ids
    for transition in transitions:
        if not isinstance(transition, dict):
            continue
        through_waypoints = transition.get("through_waypoints")
        if not isinstance(through_waypoints, list):
            continue
        waypoint_ids.update(str(waypoint_id) for waypoint_id in through_waypoints if isinstance(waypoint_id, str))
    return waypoint_ids


def format_agent_models_for_prompt(models: tuple[str, ...]) -> str:
    lines: list[str] = []
    for model in models:
        limit = AGENT_MODEL_LIMITS.get(model)
        suffix = "（数量不限）" if limit is None else f"（每个 scenario 最多 {limit} 人）"
        lines.append(f"- {model}{suffix}")
    return "\n".join(lines)


def read_prompt(args: argparse.Namespace) -> str:
    prompt_parts: list[str] = []
    if args.prompt:
        prompt_parts.append(args.prompt)
    if args.prompt_file:
        prompt_parts.append(Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip())
    return "\n\n".join(part for part in prompt_parts if part.strip())


def read_scene_prompt(scene_dir: Path, args: argparse.Namespace) -> str:
    if args.scene_prompt_file:
        path = Path(args.scene_prompt_file).expanduser().resolve()
    else:
        path = scene_dir / "scene_prompt.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_llm_prompt(
    scene_dir: Path,
    mapping: CameraRayMapping,
    scene_prompt: str,
    user_prompt: str,
    robot_intermediate_waypoints: bool = False,
    batch_agent_counts: list[int] | None = None,
) -> str:
    waypoint_context = load_waypoint_context(scene_dir)
    waypoint_text = json.dumps(waypoint_context, ensure_ascii=False, indent=2)
    agent_models_text = format_agent_models_for_prompt(AVAILABLE_AGENT_MODELS)
    if robot_intermediate_waypoints:
        robot_waypoint_rule = (
            "- 机器人使用 start_waypoint 作为起点 waypoint id；waypoints 是后续按移动顺序排列的中间点和终点 waypoint id，"
            "用于临时测试，可以包含几个中间 waypoint。"
        )
        robot_behavior_rules = """- 机器人路径必须和行人路径一起考虑，不能只单独生成；这次是临时测试，可以为机器人输出包含中间几个 waypoint 的路径。
- 机器人 waypoints 不要过密，通常 3 到 6 个即可，包含中间点和最终目标点。
- 机器人优先在开阔区域、宽走廊和宽通道中移动，尽量选择离墙、家具、柜台、桌椅、障碍物都有一定距离的 waypoint。
- 机器人半径比行人大，不能把 start_waypoint 或任何 robot waypoint 设计在狭窄位置、贴近墙边、贴近障碍物边缘或人流密集的门口。
- 机器人路线中行人可以稍微出现在机器人视野里，但不要让机器人和行人的路线产生明显碰撞或迎面冲突。
- 机器人路线可以有简单转弯，但不要设计复杂巡逻、过长路线或多段任务。"""
        robot_schema = """"robot": {
    "name": "robot",
    "from_region": {"region_id": 0, "label": "waiting_area"},
    "to_region": {"region_id": 6, "label": "consultation_room"},
    "start_waypoint": "wp_r00_waiting_area_004",
    "waypoints": ["wp_r00_waiting_area_005", "wp_r00_waiting_area_006", "wp_r00_waiting_area_007"],
    "behavior": "brief robot navigation purpose"
  }"""
    else:
        robot_waypoint_rule = (
            "- 机器人只需要选择 start_waypoint 和 goal_waypoint，不要输出中途 waypoint，不要输出 robot.waypoints。"
        )
        robot_behavior_rules = """- 机器人路径必须和行人路径一起考虑，不能只单独生成；机器人只需要 start 和 goal 表示大致方向，路线不要太长，不要太复杂。
- 机器人路径可以简单一点，但 start 和 goal 不要太近，内部规划时应想象机器人至少经过 5 个候选 waypoint 的距离后才能到达 goal；最终 JSON 仍然只输出 start_waypoint 和 goal_waypoint，不要输出中间经过点，不要设计复杂巡逻、长距离穿越或多段任务。
- 机器人优先在开阔区域、宽走廊和宽通道中移动，尽量选择离墙、家具、柜台、桌椅、障碍物都有一定距离的 waypoint。
- 机器人半径比行人大，不能把 start_waypoint 或 goal_waypoint 设计在狭窄位置、贴近墙边、贴近障碍物边缘或人流密集的门口。
- 机器人路线中行人可以稍微出现在机器人视野里，但不要让机器人和行人的路线产生明显碰撞或迎面冲突。
- 机器人 start 和 goal 应尽量在同一个连通、开阔、可安全通行的区域或宽通道内；除非用户明确要求，不要让机器人跨很多区域、穿很多门或走复杂长路线。"""
        robot_schema = """"robot": {
    "name": "robot",
    "from_region": {"region_id": 0, "label": "waiting_area"},
    "to_region": {"region_id": 6, "label": "consultation_room"},
    "start_waypoint": "wp_r00_waiting_area_004",
    "goal_waypoint": "wp_r00_waiting_area_005",
    "behavior": "brief robot navigation purpose"
  }"""

    scenario_schema = f"""{{
  "agents": [
    {{
      "name": "hunav_1",
      "role": "short semantic role",
      "model": "pedestrian",
      "from_region": {{"region_id": 0, "label": "waiting_area"}},
      "to_region": {{"region_id": 6, "label": "consultation_room"}},
      "spawn_waypoint": "wp_r00_waiting_area_001",
      "waypoints": ["wp_r00_waiting_area_002", "wp_r00_waiting_area_003"],
      "velocity": 0.8,
      "desired_velocity": 1.0,
      "behavior": "brief explanation"
    }}
  ],
  {robot_schema}
}}"""
    if batch_agent_counts:
        count_lines = "\n".join(
            f"  - scenarios[{index}] 必须恰好包含 {count} 个 agents。"
            for index, count in enumerate(batch_agent_counts)
        )
        output_requirement = f"""- 顶层必须是只包含 scenarios 数组的 JSON 对象。
- scenarios 必须恰好包含 {len(batch_agent_counts)} 个相互独立的 scenario，顺序不得改变。
{count_lines}
- scenarios 数组中的每一项都必须包含 agents 和 robot。"""
        output_schema = f"""{{
  "scenarios": [
    {scenario_schema}
  ]
}}"""
    else:
        output_requirement = "- 输出必须包含 agents 和 robot。"
        output_schema = scenario_schema

    return f"""你是室内行人和机器人轨迹生成器。请根据图片、用户需求、region 范围和候选 waypoint 列表生成完整场景轨迹，只返回合法 JSON。

图片说明：
- 图片 1 是 topdown overlay，用于理解场景语义和物体/房间位置，他是场景的俯瞰图并且将行人可以出现的位置用一层绿色的mask做了标记。
- 图片 2 是黑白 navmesh mask：白色区域可行走，黑色区域不可行走。

核心规则：
- 在设计路线前，先想象这个室内场景真实使用时是什么样的，构思好行人和机器人的路线逻辑，再开始选择 waypoint；这个思考过程只在内部完成，不要输出。
- 必须先整体规划场景：有哪些行人、机器人从哪里出发、分别要去哪里、是否会在走廊/门口/狭窄通道相遇或冲突。
- 必须先想清楚每个行人和机器人的场景语义与移动目的，例如为什么移动、从哪里出发、要去哪里。
- 行人路线需要拆成清楚的 region-to-region 分段：每一段都要明确是从哪个 region 去往哪个 region。明确起点和终点在navmesh mask中是连通的
- 对行人路线的每一段先按区域连接规则和门口/transition 规则规划，再从候选 waypoint list 中选择 waypoint id。
- 在同一个 region 内部移动时，要一个一个思考路线的下一个 waypoint，根据当前waypoint选择下一个，按照真实可能路线选择若干 waypoint，保持顺序自然、连贯。
- 每次选取下一个waypoint时需要去navmesh mask图中对比确定当前waypoint和下一个waypoint之间在navmesh中是连通的，如果不在同一片连通区域，则禁止选取这个waypoint
- 只能使用下方 waypoints_by_region 中存在的 waypoint id，严禁生成新坐标、新 id、spawn_pixel 或 waypoint_pixels。
- 行人使用 spawn_waypoint 作为起点 waypoint id；waypoints 是后续按移动顺序排列的 waypoint id，至少 2 个。
{robot_waypoint_rule}
- 通常应避免重复选择同一个 waypoint；但如果路线语义涉及返回、折返、来回移动或原路返回，可以复用已经选过的 waypoint。
- 路径必须符合真实室内移动逻辑，不能随机选点。
- 路线绝对不能穿墙，不能穿过家具、床、桌椅、柜台、货架等障碍物，也不能穿出 navmesh 可行走区域。
- 相邻两个 waypoint 的连线必须尽量位于图片 2 的白色区域内，绝对不要穿过黑色区域。
- 如果两个 waypoint 的连线会穿墙、穿过黑色区域或穿过非绿色区域，必须选择更多中间 waypoint 绕行。
- 如果进入房间或通过门口/狭窄通道，路线应在门口两侧选择连续 waypoint。
- 如果可用 waypoint JSON 里存在 region_transitions，则跨越两个 region 时必须使用对应 transition 的 through_waypoints。
- through_waypoints 表示门口/通道两侧的固定通过点，不能跳过，不能替换，必须按移动方向连续出现在路线里。
- 如果没有对应 region_transitions，才根据图像和连接图选择最合理的门两侧 waypoint。
- 多个行人分别生成独立 agent，命名为 hunav_1、hunav_2、hunav_3等。
{robot_behavior_rules}
- 机器人 start_waypoint 与最终目标 waypoint 之间的世界坐标直线距离应至少为 5 米；请选择在俯视图上明显分离的起点和终点，不要选择彼此接近的 waypoint。
- 必须输出一个 robot 对象；机器人对象的 name 必须是 "robot"；如果用户没有明确机器人数量，默认只生成 1 条机器人路径。
- 场景专用规则、区域连通规则、region_transitions 和 navmesh 约束同时适用于行人和机器人。
- role 和 behavior 简短英文即可；behavior 不要包含坐标细节。
- model 字段现在表示行人的职业类别，不是具体人物资产；具体人物模型由下层系统决定。
- model 只能是 pedestrian、doctor、police、construction_worker 之一，必须使用完全一致的小写英文值。
- 同一个 scenario 中 doctor 最多 2 人，police 最多 4 人，construction_worker 最多 4 人；pedestrian 数量不限。若人数超过某个职业上限，必须改用其他合法类别，不能突破上限。
- velocity 使用 0.6 到 1.2；desired_velocity 使用 0.8 到 1.5，且通常大于或等于 velocity。
- 尽量不要让行人会在同一时间走入同一个 waypoint，除非语义上需要交互、碰面、错身而过等，可以选取相邻waypoint表示这一语义。

输出要求：
- 只返回合法 JSON，不要 Markdown，不要解释。
- 不要输出任何像素坐标。
- 符合下面的 JSON schema：
{output_requirement}

{output_schema}

可选职业类别及数量限制：
{agent_models_text}

场景专用规则：
{scene_prompt or "无。"}

可用 region 范围和候选 waypoint 列表：
{waypoint_text}

用户需求：
{user_prompt or "请为这个室内场景生成一个规模较小且合理的行人仿真场景。"}
"""


def encode_image_part(image_path: Path) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()
    return {
        "inlineData": {
            "mimeType": "image/png",
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }


def encode_image_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{image_b64}"


def extract_json_from_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def parse_gemini_response(response_json: dict[str, Any]) -> Any:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"LLM response does not contain candidates: {response_json}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not text_chunks:
        raise ValueError(f"LLM response candidate does not contain text parts: {response_json}")
    return extract_json_from_text("\n".join(text_chunks))


def token_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def token_usage_from_dict(data: Any) -> TokenUsage:
    if not isinstance(data, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=token_count(data.get("input_tokens")),
        output_tokens=token_count(data.get("output_tokens")),
        reasoning_tokens=token_count(data.get("reasoning_tokens")),
        cached_tokens=token_count(data.get("cached_tokens")),
        total_tokens=token_count(data.get("total_tokens")),
        reported=bool(data.get("reported")),
    )


def combine_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        cached_tokens=left.cached_tokens + right.cached_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reported=left.reported or right.reported,
    )


def format_token_usage(usage: TokenUsage) -> str:
    if not usage.reported:
        return "unavailable (provider did not return token usage)"
    return (
        f"input={usage.input_tokens}, output={usage.output_tokens}, "
        f"reasoning={usage.reasoning_tokens}, cached={usage.cached_tokens}, "
        f"total={usage.total_tokens}"
    )


def parse_gemini_token_usage(response_json: dict[str, Any]) -> TokenUsage:
    usage = response_json.get("usageMetadata")
    if not isinstance(usage, dict):
        return TokenUsage()

    input_tokens = token_count(usage.get("promptTokenCount"))
    output_tokens = token_count(usage.get("candidatesTokenCount"))
    reasoning_tokens = token_count(usage.get("thoughtsTokenCount"))
    total_tokens = token_count(usage.get("totalTokenCount"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens + reasoning_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=token_count(usage.get("cachedContentTokenCount")),
        total_tokens=total_tokens,
        reported=True,
    )


def call_gemini_llm(
    config: dict[str, Any],
    prompt: str,
    image_paths: list[Path],
) -> LLMCallResult:
    api_base_url = config_value(config, "api_base_url").rstrip("/")
    api_key = config_value(config, "api_key")
    model_id = str(config.get("model_id") or config.get("model_path") or "")
    if not model_id:
        raise ValueError("LLM config is missing model_id/model_path")

    endpoint = f"{api_base_url}/models/{model_id}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}, *[encode_image_part(path) for path in image_paths]],
            }
        ],
        "generationConfig": {
            "temperature": float(config.get("temperature", 0)),
            "responseMimeType": "application/json",
        },
    }
    if config.get("max_output_tokens"):
        payload["generationConfig"]["maxOutputTokens"] = int(
            config["max_output_tokens"]
        )
    timeout = float(config.get("timeout", 180))

    try:
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        message = str(exc).replace(api_key, "<redacted-api-key>")
        raise RuntimeError(f"LLM request failed: {message}") from exc
    response_json = response.json()
    return LLMCallResult(
        data=parse_gemini_response(response_json),
        usage=parse_gemini_token_usage(response_json),
    )


def parse_openai_responses_response(response_json: dict[str, Any]) -> Any:
    text_chunks: list[str] = []
    for output_item in response_json.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                text_chunks.append(content_item["text"])

    if not text_chunks:
        raise ValueError(f"OpenAI response does not contain output text: {response_json}")
    return extract_json_from_text("\n".join(text_chunks))


def parse_openai_token_usage(response_json: dict[str, Any]) -> TokenUsage:
    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()

    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    cached_tokens = (
        token_count(input_details.get("cached_tokens"))
        if isinstance(input_details, dict)
        else 0
    )
    reasoning_tokens = (
        token_count(output_details.get("reasoning_tokens"))
        if isinstance(output_details, dict)
        else 0
    )
    input_tokens = token_count(usage.get("input_tokens"))
    output_tokens = token_count(usage.get("output_tokens"))
    total_tokens = token_count(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        reported=True,
    )


def call_openai_responses_llm(
    config: dict[str, Any],
    prompt: str,
    image_paths: list[Path],
) -> LLMCallResult:
    api_base_url = config_value(config, "api_base_url").rstrip("/")
    api_key = config_value(config, "api_key")
    model_id = config_value(config, "model_id")
    endpoint = f"{api_base_url}/responses"
    image_detail = str(config.get("image_detail", "high"))

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend(
        {
            "type": "input_image",
            "image_url": encode_image_data_url(path),
            "detail": image_detail,
        }
        for path in image_paths
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "input": [{"role": "user", "content": content}],
    }
    if config.get("temperature") is not None:
        payload["temperature"] = float(config["temperature"])
    if config.get("max_output_tokens"):
        payload["max_output_tokens"] = int(config["max_output_tokens"])
    if config.get("reasoning_effort"):
        payload["reasoning"] = {"effort": str(config["reasoning_effort"])}

    timeout = float(config.get("timeout", 180))
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        message = str(exc).replace(api_key, "<redacted-api-key>")
        raise RuntimeError(f"OpenAI LLM request failed: {message}") from exc
    response_json = response.json()
    return LLMCallResult(
        data=parse_openai_responses_response(response_json),
        usage=parse_openai_token_usage(response_json),
    )


def call_llm_with_usage(
    config: dict[str, Any],
    prompt: str,
    image_paths: list[Path],
) -> LLMCallResult:
    llm_type = str(config.get("type", "gemini")).lower()
    if llm_type in {"gemini", "google"}:
        return call_gemini_llm(config, prompt, image_paths)
    if llm_type in {"openai", "openai_responses", "responses"}:
        return call_openai_responses_llm(config, prompt, image_paths)
    raise ValueError(f"Unsupported LLM config type: {llm_type}")


def call_llm(config: dict[str, Any], prompt: str, image_paths: list[Path]) -> Any:
    """Compatibility wrapper returning only parsed model data."""
    return call_llm_with_usage(config, prompt, image_paths).data


def extract_pixel(value: Any, field_name: str) -> tuple[float, float]:
    if isinstance(value, dict):
        if "pixel" in value:
            return extract_pixel(value["pixel"], field_name)
        if "point" in value:
            return extract_pixel(value["point"], field_name)
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])

    raise ValueError(f"{field_name} must be [x, y], {{'x': x, 'y': y}}, or {{'pixel': [x, y]}}")


def extract_waypoint_id(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("id", "waypoint", "waypoint_id"):
            if isinstance(value.get(key), str):
                return value[key]
    raise ValueError(f"{field_name} must be a waypoint id string")


def pixel_from_waypoint_id(
    waypoint_id: str,
    waypoint_lookup: dict[str, tuple[float, float]],
    field_name: str,
) -> tuple[float, float]:
    if waypoint_id not in waypoint_lookup:
        raise ValueError(f"{field_name} references unknown waypoint id: {waypoint_id}")
    return waypoint_lookup[waypoint_id]


def extract_spawn_pixel(
    agent: dict[str, Any],
    agent_name: str,
    waypoint_lookup: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    if waypoint_lookup is not None:
        for key in ("spawn_waypoint", "spawn_waypoint_id", "spawn_id", "start_waypoint"):
            if key in agent:
                waypoint_id = extract_waypoint_id(agent[key], f"{agent_name}.{key}")
                return pixel_from_waypoint_id(waypoint_id, waypoint_lookup, f"{agent_name}.{key}")

    for key in ("spawn_pixel", "spawn", "pose_pixel", "start_pixel", "start"):
        if key in agent:
            return extract_pixel(agent[key], f"{agent_name}.{key}")
    raise ValueError(f"{agent_name} is missing spawn_pixel")


def extract_waypoint_pixels(
    agent: dict[str, Any],
    agent_name: str,
    waypoint_lookup: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    raw_waypoints = None
    if waypoint_lookup is not None:
        for key in ("waypoints", "waypoint_ids", "path_waypoints", "route_waypoints", "path"):
            if key in agent:
                raw_waypoints = agent[key]
                break

        if raw_waypoints is not None:
            if not isinstance(raw_waypoints, list) or not raw_waypoints:
                raise ValueError(f"{agent_name} must contain a non-empty waypoint id array")
            return [
                pixel_from_waypoint_id(
                    extract_waypoint_id(waypoint, f"{agent_name}.{key}[{index}]"),
                    waypoint_lookup,
                    f"{agent_name}.{key}[{index}]",
                )
                for index, waypoint in enumerate(raw_waypoints)
            ]

    for key in ("waypoint_pixels", "waypoints_pixel", "waypoints", "path_pixels", "path"):
        if key in agent:
            raw_waypoints = agent[key]
            break

    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError(f"{agent_name} must contain a non-empty waypoint_pixels array")

    return [
        extract_pixel(waypoint, f"{agent_name}.waypoint_pixels[{index}]")
        for index, waypoint in enumerate(raw_waypoints)
    ]


def validate_pixel(mapping: CameraRayMapping, pixel: tuple[float, float], field_name: str) -> None:
    x, y = pixel
    if not (0.0 <= x < mapping.image_width and 0.0 <= y < mapping.image_height):
        raise ValueError(
            f"{field_name} pixel ({x}, {y}) is outside image bounds "
            f"0 <= x < {mapping.image_width}, 0 <= y < {mapping.image_height}"
        )


def yaw_degrees(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def rounded(value: float, places: int) -> float:
    return round(float(value), places)


def route_from_agent(
    agent: dict[str, Any],
    name: str,
    mapping: CameraRayMapping,
    args: argparse.Namespace,
    waypoint_lookup: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[float], list[list[float]]]:
    spawn_pixel = extract_spawn_pixel(agent, name, waypoint_lookup)
    waypoint_pixels = extract_waypoint_pixels(agent, name, waypoint_lookup)

    validate_pixel(mapping, spawn_pixel, f"{name}.spawn_pixel")
    for waypoint_index, waypoint_pixel in enumerate(waypoint_pixels):
        validate_pixel(mapping, waypoint_pixel, f"{name}.waypoint_pixels[{waypoint_index}]")

    spawn_world = mapping.pixel_to_world(*spawn_pixel)
    waypoint_worlds = [mapping.pixel_to_world(*pixel) for pixel in waypoint_pixels]
    route_points = [spawn_world, *waypoint_worlds]
    route_yaws = [
        yaw_degrees(route_points[i], route_points[i + 1])
        for i in range(len(route_points) - 1)
    ]
    last_yaw = route_yaws[-1] if route_yaws else 0.0

    spawn_yaw = float(agent.get("yaw", agent.get("spawn_yaw", route_yaws[0] if route_yaws else 0.0)))
    waypoint_yaws = [*route_yaws[1:], last_yaw]
    pose = [
        rounded(spawn_world[0], args.round),
        rounded(spawn_world[1], args.round),
        rounded(spawn_yaw, args.round),
    ]
    waypoints = [
        [
            rounded(world_xy[0], args.round),
            rounded(world_xy[1], args.round),
            rounded(waypoint_yaws[waypoint_index], args.round),
        ]
        for waypoint_index, world_xy in enumerate(waypoint_worlds)
    ]
    return pose, waypoints


def robot_from_data(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    robot = data.get("robot")
    if isinstance(robot, dict):
        return robot
    robots = data.get("robots")
    if isinstance(robots, list) and robots and isinstance(robots[0], dict):
        return robots[0]
    return None


def robot_route_ids(robot: dict[str, Any]) -> tuple[str, list[str]] | None:
    start_id: str | None = None
    for key in ("start_waypoint", "start_waypoint_id", "spawn_waypoint", "spawn_waypoint_id", "spawn_id"):
        if key in robot:
            start_id = extract_waypoint_id(robot[key], f"robot.{key}")
            break
    if start_id is None:
        return None

    for key in ("goal_waypoint", "goal_waypoint_id", "goal_id", "target_waypoint", "target_waypoint_id"):
        if key in robot:
            return start_id, [extract_waypoint_id(robot[key], f"robot.{key}")]

    raw_waypoints = None
    for key in ("waypoints", "waypoint_ids", "path_waypoints", "route_waypoints", "path"):
        if key in robot:
            raw_waypoints = robot[key]
            break
    if not isinstance(raw_waypoints, list):
        return None
    waypoint_ids = [
        extract_waypoint_id(waypoint, f"robot.waypoints[{index}]")
        for index, waypoint in enumerate(raw_waypoints)
    ]
    return start_id, waypoint_ids


def normalize_robot_agent(
    robot: dict[str, Any],
    keep_intermediate_waypoints: bool = False,
) -> dict[str, Any]:
    route = robot_route_ids(robot)
    if route is None:
        return robot
    start_id, waypoint_ids = route
    route_ids = [start_id, *waypoint_ids]
    if len(route_ids) < 2:
        return robot

    sparse_robot = dict(robot)
    for key in (
        "spawn_waypoint",
        "spawn_waypoint_id",
        "spawn_id",
        "start_waypoint_id",
        "goal_waypoint_id",
        "goal_id",
        "target_waypoint",
        "target_waypoint_id",
        "waypoint_ids",
        "path_waypoints",
        "route_waypoints",
        "path",
    ):
        sparse_robot.pop(key, None)
    sparse_robot["name"] = "robot"
    sparse_robot["start_waypoint"] = route_ids[0]
    sparse_robot["goal_waypoint"] = route_ids[-1]
    sparse_robot["waypoints"] = route_ids[1:] if keep_intermediate_waypoints else [route_ids[-1]]
    return sparse_robot


def build_robot_entry(
    robot: dict[str, Any],
    mapping: CameraRayMapping,
    args: argparse.Namespace,
    waypoint_lookup: dict[str, tuple[float, float]] | None = None,
    waypoint_context: dict[str, Any] | None = None,
    keep_intermediate_waypoints: bool = False,
) -> dict[str, Any]:
    robot = normalize_robot_agent(robot, keep_intermediate_waypoints)
    pose, waypoints = route_from_agent(robot, "robot", mapping, args, waypoint_lookup)
    return {
        "name": "robot",
        "pose": pose,
        "waypoints": waypoints,
        "behavior": str(robot.get("behavior") or ""),
    }


def validated_agent_models(
    agents: list[dict[str, Any]],
    default_model: str,
) -> list[str]:
    models: list[str] = []
    counts = {model: 0 for model in AVAILABLE_AGENT_MODELS}

    for index, agent in enumerate(agents, start=1):
        name = str(agent.get("name") or f"hunav_{index}")
        raw_model = agent.get("model") or default_model
        if not isinstance(raw_model, str) or raw_model not in AVAILABLE_AGENT_MODELS:
            allowed = ", ".join(AVAILABLE_AGENT_MODELS)
            raise ValueError(
                f"{name}.model must be one of: {allowed}; got {raw_model!r}"
            )
        models.append(raw_model)
        counts[raw_model] += 1

    for model, limit in AGENT_MODEL_LIMITS.items():
        if counts[model] > limit:
            raise ValueError(
                f"scenario contains {counts[model]} {model} agents; maximum is {limit}"
            )
    return models


def build_scenario(
    agents: list[dict[str, Any]],
    mapping: CameraRayMapping,
    args: argparse.Namespace,
    waypoint_lookup: dict[str, tuple[float, float]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    dynamic_agents: list[dict[str, Any]] = []
    agent_models = validated_agent_models(agents, args.default_model)

    for index, (agent, agent_model) in enumerate(zip(agents, agent_models), start=1):
        name = str(agent.get("name") or f"hunav_{index}")
        pose, waypoints = route_from_agent(agent, name, mapping, args, waypoint_lookup)

        dynamic_agents.append(
            {
                "name": name,
                "model": agent_model,
                "pose": pose,
                "behavior_tree": str(agent.get("behavior_tree") or args.default_behavior_tree),
                "velocity": rounded(float(agent.get("velocity", args.default_velocity)), args.round),
                "desired_velocity": rounded(
                    float(agent.get("desired_velocity", args.default_desired_velocity)),
                    args.round,
                ),
                "waypoints": waypoints,
            }
        )

    return {"dynamic": dynamic_agents}


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    mapping_path = (
        Path(args.mapping).expanduser().resolve()
        if args.mapping
        else scene_dir / "topdown_mapping.json"
    )
    output_path = Path(args.output).expanduser().resolve()

    mapping = load_mapping(mapping_path)
    llm_config_path = Path(args.llm_config).expanduser().resolve()
    llm_config = load_yaml_file(llm_config_path)
    overlay_image_path = scene_dir / "overlay.png"
    if not overlay_image_path.exists():
        raise FileNotFoundError(f"Missing required overlay image: {overlay_image_path}")
    navmesh_image_path = scene_dir / "navmesh_mask.png"
    if not navmesh_image_path.exists():
        raise FileNotFoundError(f"Missing required navmesh mask image: {navmesh_image_path}")
    llm_image_paths = [overlay_image_path, navmesh_image_path]

    waypoint_context = load_waypoint_context(scene_dir)
    waypoint_lookup = build_waypoint_lookup(waypoint_context)
    prompt = build_llm_prompt(
        scene_dir,
        mapping,
        read_scene_prompt(scene_dir, args),
        read_prompt(args),
        args.robot_intermediate_waypoints,
    )
    if args.dry_run_prompt:
        print(prompt)
        print("\nAttached images:")
        for image_index, image_path in enumerate(llm_image_paths, start=1):
            print(f"{image_index}. {image_path}")
        return

    llm_result = call_llm_with_usage(llm_config, prompt, llm_image_paths)
    llm_data = llm_result.data
    print(
        TOKEN_USAGE_PREFIX
        + json.dumps(llm_result.usage.to_dict(), separators=(",", ":")),
        flush=True,
    )

    if args.save_llm_output:
        save_path = Path(args.save_llm_output).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(llm_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    agents = agents_from_data(llm_data)

    scenario = build_scenario(agents, mapping, args, waypoint_lookup)
    robot = robot_from_data(llm_data)
    if robot is not None:
        scenario["robot"] = build_robot_entry(
            robot,
            mapping,
            args,
            waypoint_lookup,
            waypoint_context,
            args.robot_intermediate_waypoints,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Generated from topdown pixel waypoints\n\n"
        + yaml.dump(
            scenario,
            Dumper=ScenarioYamlDumper,
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(scenario['dynamic'])} agents to {output_path}")


if __name__ == "__main__":
    main()
