#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from generate_scenario_from_llm import (
    AVAILABLE_AGENT_MODELS,
    DEFAULT_BEHAVIOR_TREE,
    DEFAULT_DESIRED_VELOCITY,
    DEFAULT_MODEL,
    DEFAULT_VELOCITY,
    ScenarioYamlDumper,
    agents_from_data,
    build_scenario,
    build_robot_entry,
    build_waypoint_lookup,
    load_waypoint_context,
    load_mapping,
    robot_from_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert saved LLM waypoint JSON into a HuNav scenario YAML.")
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing topdown_mapping.json and waypoints_by_region.json.",
    )
    parser.add_argument(
        "--llm-agents",
        required=True,
        help="Saved LLM JSON file containing an agents array.",
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
    parser.add_argument(
        "--robot-intermediate-waypoints",
        action="store_true",
        help="Keep robot intermediate waypoints from the saved LLM JSON instead of reducing robot to start and goal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    mapping_path = (
        Path(args.mapping).expanduser().resolve()
        if args.mapping
        else scene_dir / "topdown_mapping.json"
    )
    llm_agents_path = Path(args.llm_agents).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    mapping = load_mapping(mapping_path)
    waypoint_context = load_waypoint_context(scene_dir)
    waypoint_lookup = build_waypoint_lookup(waypoint_context)
    llm_data = json.loads(llm_agents_path.read_text(encoding="utf-8"))
    scenario = build_scenario(agents_from_data(llm_data), mapping, args, waypoint_lookup)
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
        "# Generated from saved LLM waypoints\n\n"
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
