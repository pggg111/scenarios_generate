#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_scenario_from_llm import (
    DEFAULT_MODEL,
    TOKEN_USAGE_PREFIX,
    TokenUsage,
    combine_token_usage,
    format_token_usage,
    token_usage_from_dict,
    validated_agent_models,
)


REQUIRED_SCENE_FILES = (
    "overlay.png",
    "navmesh_mask.png",
    "topdown_mapping.json",
    "waypoints_by_region.json",
    "scene_prompt.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple distinct trajectory scenarios for one prepared scene."
    )
    parser.add_argument("--scene-dir", required=True, help="Prepared scene directory.")
    parser.add_argument("--output-root", required=True, help="Root directory for all generated outputs.")
    parser.add_argument("--llm-config", required=True, help="LLM YAML configuration path.")
    parser.add_argument("--prompt", required=True, help="Common scenario request for every variant.")
    parser.add_argument("--count", type=int, default=5, help="Number of distinct variants. Default: 5.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Scenarios requested in each LLM call. Default: 10; use 1 for legacy mode.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum generation attempts per variant. Default: 3.",
    )
    parser.add_argument(
        "--pedestrians",
        type=int,
        default=None,
        help=(
            "Fixed number of agents for every scenario. Default: 2 when no "
            "min/max range is provided."
        ),
    )
    parser.add_argument(
        "--min-pedestrians",
        type=int,
        help="Minimum agent count for a deterministic balanced schedule.",
    )
    parser.add_argument(
        "--max-pedestrians",
        type=int,
        help="Maximum agent count for a deterministic balanced schedule.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260930,
        help="Seed used to shuffle a ranged agent-count schedule.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=12,
        help=(
            "Number of recent accepted route summaries included in the next LLM prompt. "
            "Default: 12."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate variants whose output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned variant output paths without calling the LLM.",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Do not render per-scenario previews or the combined overview.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Render individual previews but do not create the combined overview image.",
    )
    return parser.parse_args()


def build_agent_count_schedule(
    count: int,
    fixed_count: int | None,
    minimum: int | None,
    maximum: int | None,
    seed: int,
) -> list[int]:
    """Return stable per-scenario counts, balanced across an inclusive range."""
    if minimum is None and maximum is None:
        selected = 2 if fixed_count is None else fixed_count
        if selected < 1:
            raise ValueError("--pedestrians must be at least 1")
        return [selected] * count

    if fixed_count is not None:
        raise ValueError(
            "use either --pedestrians or --min-pedestrians/--max-pedestrians, not both"
        )
    if minimum is None or maximum is None:
        raise ValueError("--min-pedestrians and --max-pedestrians must be used together")
    if minimum < 1:
        raise ValueError("--min-pedestrians must be at least 1")
    if maximum < minimum:
        raise ValueError("--max-pedestrians must be greater than or equal to the minimum")

    choices = list(range(minimum, maximum + 1))
    schedule = [choices[index % len(choices)] for index in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def waypoint_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("id", "waypoint_id", "name"):
            if isinstance(value.get(key), str):
                return value[key]
    return str(value)


def route_from_agent(agent: dict[str, Any]) -> tuple[str, ...]:
    start = agent.get("spawn_waypoint") or agent.get("start_waypoint") or agent.get("spawn")
    route = [waypoint_id(start)] if start is not None else []
    raw_waypoints = agent.get("waypoints") or agent.get("waypoint_ids") or []
    if isinstance(raw_waypoints, list):
        route.extend(waypoint_id(value) for value in raw_waypoints)
    return tuple(route)


def route_from_robot(robot: dict[str, Any]) -> tuple[str, ...]:
    start = robot.get("start_waypoint") or robot.get("spawn_waypoint") or robot.get("start")
    route = [waypoint_id(start)] if start is not None else []
    raw_waypoints = robot.get("waypoints")
    if isinstance(raw_waypoints, list) and raw_waypoints:
        route.extend(waypoint_id(value) for value in raw_waypoints)
    else:
        goal = robot.get("goal_waypoint") or robot.get("goal")
        if goal is not None:
            route.append(waypoint_id(goal))
    return tuple(route)


def scenario_routes(data: Any, expected_pedestrians: int) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    agents = data.get("agents")
    if not isinstance(agents, list) or len(agents) != expected_pedestrians:
        actual = len(agents) if isinstance(agents, list) else 0
        raise ValueError(f"expected {expected_pedestrians} agents, got {actual}")
    if not all(isinstance(agent, dict) for agent in agents):
        raise ValueError("every agent must be a JSON object")
    validated_agent_models(agents, DEFAULT_MODEL)
    robot = data.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("missing robot object")

    agent_routes = [route_from_agent(agent) for agent in agents]
    robot_route = route_from_robot(robot)
    if any(len(route) < 3 for route in agent_routes):
        raise ValueError("each pedestrian route must contain a spawn and at least two waypoints")
    if len(robot_route) < 2:
        raise ValueError("robot route must contain start and goal waypoints")
    return agent_routes, robot_route


def scenario_signature(
    agent_routes: list[tuple[str, ...]],
    robot_route: tuple[str, ...],
) -> str:
    # Sorting makes a simple hunav_1/hunav_2 name swap count as the same scenario.
    normalized = {"agents": sorted(agent_routes), "robot": robot_route}
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def route_summary(
    variant_number: int,
    agent_routes: list[tuple[str, ...]],
    robot_route: tuple[str, ...],
) -> str:
    parts = [
        f"候选 {variant_number} 行人{i}: {' -> '.join(route)}"
        for i, route in enumerate(agent_routes, start=1)
    ]
    parts.append(f"候选 {variant_number} 机器人: {' -> '.join(robot_route)}")
    return "\n".join(parts)


def variant_prompt(
    common_prompt: str,
    variant_number: int,
    count: int,
    pedestrians: int,
    previous_summaries: list[str],
    attempt: int,
) -> str:
    exclusion = ""
    if previous_summaries:
        exclusion = (
            "\n以下路线已经用于先前候选，本次不得生成完全相同的行人和机器人路线组合：\n"
            + "\n".join(previous_summaries)
        )
    retry_rule = ""
    if attempt > 1:
        retry_rule = f"\n这是该候选的第 {attempt} 次尝试，请明显调整起点、终点或主要路线。"
    return f"""{common_prompt}

这是同一场景的第 {variant_number}/{count} 个独立候选方案。
必须恰好生成 {pedestrians} 个行人和 1 个机器人。
优先通过不同的起点、目标区域和 waypoint 路线形成有意义的差异；角色、模型或速度变化不能作为唯一差异。
仍须严格遵守场景连通规则、region_transitions 和 navmesh 约束。{exclusion}{retry_rule}"""


def batch_variant_prompt(
    common_prompt: str,
    items: list[tuple[int, int]],
    total_count: int,
    previous_summaries: list[str],
    attempt: int,
) -> str:
    item_lines = "\n".join(
        f"- scenarios[{index}] 对应 scenario_{variant_number:03d}，"
        f"必须恰好生成 {agent_count} 个 agents 和 1 个机器人。"
        for index, (variant_number, agent_count) in enumerate(items)
    )
    exclusion = ""
    if previous_summaries:
        exclusion = (
            "\n以下路线已经用于其他候选，本批不得生成完全相同的行人和机器人路线组合：\n"
            + "\n".join(previous_summaries)
        )
    retry_rule = ""
    if attempt > 1:
        retry_rule = (
            f"\n这是本批的第 {attempt} 次尝试，请明显调整各条 scenario 的"
            "起点、终点或主要路线。"
        )
    return f"""{common_prompt}

这次需要一次生成 {len(items)} 个相互独立的 scenario；整个场景计划共 {total_count} 条。
请严格保持下面的数组顺序和人数：
{item_lines}
同一批内各条 scenario 的完整行人和机器人 waypoint 路线组合必须互不相同。
同一批内不要复用完全相同的 robot start_waypoint 到 goal_waypoint 组合；机器人移动方向和距离也应尽量有变化，不要让所有路线都刚好接近 5 米。
在场景连通和安全允许的前提下，行人的主要 waypoint 路线也应尽量避免在批内重复。
优先通过不同的起点、目标区域和 waypoint 路线形成有意义的差异；角色、职业或速度变化不能作为唯一差异。
仍须严格遵守场景连通规则、region_transitions 和 navmesh 约束。{exclusion}{retry_rule}"""


def load_existing_variant(
    llm_output: Path,
    expected_pedestrians: int,
) -> tuple[list[tuple[str, ...]], tuple[str, ...]]:
    data = json.loads(llm_output.read_text(encoding="utf-8"))
    return scenario_routes(data, expected_pedestrians)


def extract_token_usage(output: str) -> tuple[list[TokenUsage], str]:
    usages: list[TokenUsage] = []
    visible_lines: list[str] = []
    for line in output.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(TOKEN_USAGE_PREFIX):
            try:
                payload = json.loads(stripped[len(TOKEN_USAGE_PREFIX) :])
            except json.JSONDecodeError:
                visible_lines.append(line)
                continue
            usages.append(token_usage_from_dict(payload))
        else:
            visible_lines.append(line)
    return usages, "".join(visible_lines)


def print_captured_output(stdout: str, stderr: str) -> list[TokenUsage]:
    usages, visible_stdout = extract_token_usage(stdout)
    if visible_stdout:
        print(visible_stdout, end="" if visible_stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    return usages


def token_usage_suffix(api_calls: int, unreported_calls: int) -> str:
    suffix = f"api_calls={api_calls}"
    if unreported_calls:
        suffix += f", usage_unreported_calls={unreported_calls}"
    return suffix


def print_total_token_usage(
    usage: TokenUsage,
    api_calls: int,
    unreported_calls: int,
) -> None:
    if api_calls == 0:
        details = "input=0, output=0, reasoning=0, cached=0, total=0"
    else:
        details = format_token_usage(usage)
    print(
        f"TOTAL TOKENS THIS RUN: {details}; "
        f"{token_usage_suffix(api_calls, unreported_calls)}",
        flush=True,
    )


def render_scenario_preview(
    renderer: Path,
    scene_dir: Path,
    scenario_path: Path,
    preview_path: Path,
) -> None:
    command = [
        sys.executable,
        str(renderer),
        "--mapping",
        str(scene_dir / "topdown_mapping.json"),
        "--view",
        "overlay",
        "--scenario",
        str(scenario_path),
        "--save-overlay",
        str(preview_path),
        "--no-show",
    ]
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    environment.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to render preview for {scenario_path}")


def create_overview(preview_paths: list[Path], output_path: Path) -> None:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    thumbnails: list[Image.Image] = []
    for preview_path in preview_paths:
        with Image.open(preview_path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((800, 800), resampling)
            thumbnails.append(thumbnail)

    columns = min(2, len(thumbnails))
    rows = math.ceil(len(thumbnails) / columns)
    margin = 12
    label_height = 32
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails) + label_height
    overview = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(overview)
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        cell_x = margin + column * (cell_width + margin)
        cell_y = margin + row * (cell_height + margin)
        image_x = cell_x + (cell_width - thumbnail.width) // 2
        image_y = cell_y + label_height
        draw.text((cell_x + 6, cell_y + 8), f"scenario_{index + 1:03d}", fill="black")
        overview.paste(thumbnail, (image_x, image_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(output_path)


def remove_rejected_batch_outputs(
    scene_output_root: Path,
    batch_items: list[tuple[int, int]],
) -> None:
    for variant_number, _ in batch_items:
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        for filename in ("llm_agents.json", "generated_scenario.yaml"):
            path = variant_dir / filename
            if path.is_file():
                path.unlink()


def run_batched(
    args: argparse.Namespace,
    agent_count_schedule: list[int],
) -> None:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    llm_config = Path(args.llm_config).expanduser().resolve()
    batch_generator = (
        Path(__file__).with_name("generate_scenario_batch_from_llm.py").resolve()
    )
    renderer = Path(__file__).with_name("topdown_click_to_world.py").resolve()

    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    missing = [name for name in REQUIRED_SCENE_FILES if not (scene_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Scene is not ready; missing: {', '.join(missing)}")
    if not llm_config.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")
    if not batch_generator.is_file():
        raise FileNotFoundError(f"Batch generator not found: {batch_generator}")

    scene_output_root = output_root / scene_dir.name
    signatures: set[str] = set()
    summaries: list[str] = []
    pending: list[tuple[int, int]] = []
    generated = 0
    existing = 0
    total_usage = TokenUsage()
    total_api_calls = 0
    total_unreported_calls = 0

    # Validate all resumable outputs first, then batch only the missing variants.
    for variant_number, expected_agents in enumerate(agent_count_schedule, start=1):
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        llm_output = variant_dir / "llm_agents.json"
        scenario_output = variant_dir / "generated_scenario.yaml"
        if (
            not args.overwrite
            and llm_output.is_file()
            and scenario_output.is_file()
        ):
            try:
                agent_routes, robot_route = load_existing_variant(
                    llm_output, expected_agents
                )
                signature = scenario_signature(agent_routes, robot_route)
                if signature in signatures:
                    raise ValueError("duplicate existing waypoint route combination")
                signatures.add(signature)
                summaries.append(
                    route_summary(variant_number, agent_routes, robot_route)
                )
                existing += 1
                print(f"EXISTS scenario_{variant_number:03d}")
                print(
                    f"TOKENS scenario_{variant_number:03d}: 0 "
                    "(existing output; no API call this run)"
                )
                continue
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(
                    f"INVALID EXISTING scenario_{variant_number:03d}: {exc}; "
                    "use --overwrite"
                )
                raise SystemExit(1) from exc
        pending.append((variant_number, expected_agents))

    if args.dry_run:
        for offset in range(0, len(pending), args.batch_size):
            batch_items = pending[offset : offset + args.batch_size]
            numbers = ",".join(
                f"{variant_number:03d}" for variant_number, _ in batch_items
            )
            print(f"READY BATCH [{numbers}]")
        print(
            f"Summary: requested={args.count}, pending={len(pending)}, existing={existing}, "
            f"batch_size={args.batch_size}"
        )
        return

    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    total_batches = math.ceil(len(pending) / args.batch_size)
    for offset in range(0, len(pending), args.batch_size):
        batch_items = pending[offset : offset + args.batch_size]
        batch_index = offset // args.batch_size + 1
        first_number = batch_items[0][0]
        last_number = batch_items[-1][0]
        batch_label = f"batch_{first_number:03d}_{last_number:03d}"
        accepted = False
        batch_usage = TokenUsage()
        batch_api_calls = 0
        batch_unreported_calls = 0

        for attempt in range(1, args.max_attempts + 1):
            prompt = batch_variant_prompt(
                args.prompt,
                batch_items,
                args.count,
                summaries[-args.history_window :] if args.history_window else [],
                attempt,
            )
            batch_payload = [
                {
                    "variant_number": variant_number,
                    "agent_count": agent_count,
                }
                for variant_number, agent_count in batch_items
            ]
            command = [
                sys.executable,
                str(batch_generator),
                "--scene-dir",
                str(scene_dir),
                "--output-root",
                str(scene_output_root),
                "--llm-config",
                str(llm_config),
                "--prompt",
                prompt,
                "--batch-items",
                json.dumps(batch_payload, separators=(",", ":")),
            ]
            print(
                f"GENERATING {batch_label} | batch {batch_index}/{total_batches} | "
                f"{len(batch_items)} scenarios | "
                f"attempt {attempt}/{args.max_attempts}",
                flush=True,
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )
            attempt_usages = print_captured_output(result.stdout, result.stderr)
            for attempt_usage in attempt_usages:
                batch_usage = combine_token_usage(batch_usage, attempt_usage)
                total_usage = combine_token_usage(total_usage, attempt_usage)
                batch_api_calls += 1
                total_api_calls += 1
                if not attempt_usage.reported:
                    batch_unreported_calls += 1
                    total_unreported_calls += 1
            if result.returncode != 0:
                print(f"Batch attempt failed with exit code {result.returncode}", flush=True)
                continue

            try:
                accepted_items: list[
                    tuple[int, list[tuple[str, ...]], tuple[str, ...], str]
                ] = []
                batch_signatures: set[str] = set()
                for variant_number, expected_agents in batch_items:
                    llm_output = (
                        scene_output_root
                        / f"scenario_{variant_number:03d}"
                        / "llm_agents.json"
                    )
                    agent_routes, robot_route = load_existing_variant(
                        llm_output, expected_agents
                    )
                    signature = scenario_signature(agent_routes, robot_route)
                    if signature in signatures or signature in batch_signatures:
                        raise ValueError(
                            f"scenario_{variant_number:03d} duplicates an existing "
                            "waypoint route combination"
                        )
                    batch_signatures.add(signature)
                    accepted_items.append(
                        (variant_number, agent_routes, robot_route, signature)
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Rejected batch output: {exc}", flush=True)
                remove_rejected_batch_outputs(scene_output_root, batch_items)
                continue

            for variant_number, agent_routes, robot_route, signature in accepted_items:
                signatures.add(signature)
                summaries.append(
                    route_summary(variant_number, agent_routes, robot_route)
                )
                print(f"GENERATED scenario_{variant_number:03d}", flush=True)
            generated += len(accepted_items)
            accepted = True
            completed_count = existing + generated
            print(
                f"PROGRESS: completed={completed_count}/{args.count} "
                f"({completed_count / args.count:.1%})",
                flush=True,
            )
            print(
                f"TOKENS {batch_label}: {format_token_usage(batch_usage)}; "
                f"{token_usage_suffix(batch_api_calls, batch_unreported_calls)}",
                flush=True,
            )
            if batch_api_calls:
                print(
                    f"TOKEN AVERAGE {batch_label}: "
                    f"total={batch_usage.total_tokens / len(batch_items):.1f} "
                    f"per scenario ({len(batch_items)} scenarios)",
                    flush=True,
                )
            break

        if not accepted:
            print(
                f"FAILED {batch_label}: no valid distinct batch after "
                f"{args.max_attempts} attempts",
                flush=True,
            )
            print(
                f"TOKENS {batch_label}: {format_token_usage(batch_usage)}; "
                f"{token_usage_suffix(batch_api_calls, batch_unreported_calls)}",
                flush=True,
            )
            print_total_token_usage(
                total_usage, total_api_calls, total_unreported_calls
            )
            raise SystemExit(1)

    if args.skip_previews:
        print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
        print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)
        return

    preview_paths: list[Path] = []
    for variant_number in range(1, args.count + 1):
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        scenario_path = variant_dir / "generated_scenario.yaml"
        preview_path = variant_dir / "scenario_preview.png"
        if preview_path.is_file() and not args.overwrite:
            print(f"EXISTS PREVIEW scenario_{variant_number:03d}", flush=True)
            preview_paths.append(preview_path)
            continue
        print(f"RENDERING scenario_{variant_number:03d}", flush=True)
        render_scenario_preview(renderer, scene_dir, scenario_path, preview_path)
        preview_paths.append(preview_path)

    if args.skip_overview:
        print("SKIPPED OVERVIEW (individual previews were generated)", flush=True)
        print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
        print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)
        return

    overview_path = scene_output_root / "scenarios_overview.png"
    create_overview(preview_paths, overview_path)
    print(f"WROTE OVERVIEW {overview_path}", flush=True)
    print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
    print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.history_window < 0:
        raise ValueError("--history-window cannot be negative")
    agent_count_schedule = build_agent_count_schedule(
        args.count,
        args.pedestrians,
        args.min_pedestrians,
        args.max_pedestrians,
        args.seed,
    )
    if args.batch_size > 1:
        run_batched(args, agent_count_schedule)
        return

    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    llm_config = Path(args.llm_config).expanduser().resolve()
    generator = Path(__file__).with_name("generate_scenario_from_llm.py").resolve()
    renderer = Path(__file__).with_name("topdown_click_to_world.py").resolve()

    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    missing = [name for name in REQUIRED_SCENE_FILES if not (scene_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Scene is not ready; missing: {', '.join(missing)}")
    if not llm_config.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")

    scene_output_root = output_root / scene_dir.name
    signatures: set[str] = set()
    summaries: list[str] = []
    generated = 0
    existing = 0
    total_usage = TokenUsage()
    total_api_calls = 0
    total_unreported_calls = 0

    for variant_number in range(1, args.count + 1):
        expected_agents = agent_count_schedule[variant_number - 1]
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        llm_output = variant_dir / "llm_agents.json"
        scenario_output = variant_dir / "generated_scenario.yaml"

        if not args.overwrite and llm_output.is_file() and scenario_output.is_file():
            try:
                agent_routes, robot_route = load_existing_variant(llm_output, expected_agents)
                signature = scenario_signature(agent_routes, robot_route)
                if signature in signatures:
                    print(f"DUPLICATE EXISTING scenario_{variant_number:03d}; use --overwrite")
                    raise SystemExit(1)
                signatures.add(signature)
                summaries.append(route_summary(variant_number, agent_routes, robot_route))
                existing += 1
                print(f"EXISTS scenario_{variant_number:03d}")
                print(
                    f"TOKENS scenario_{variant_number:03d}: 0 "
                    "(existing output; no API call this run)"
                )
                continue
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"INVALID EXISTING scenario_{variant_number:03d}: {exc}; use --overwrite")
                raise SystemExit(1) from exc

        if args.dry_run:
            print(f"READY scenario_{variant_number:03d}: {variant_dir}")
            continue

        accepted = False
        scenario_usage = TokenUsage()
        scenario_api_calls = 0
        scenario_unreported_calls = 0
        for attempt in range(1, args.max_attempts + 1):
            prompt = variant_prompt(
                args.prompt,
                variant_number,
                args.count,
                expected_agents,
                summaries[-args.history_window :] if args.history_window else [],
                attempt,
            )
            command = [
                sys.executable,
                str(generator),
                "--scene-dir",
                str(scene_dir),
                "--llm-config",
                str(llm_config),
                "--prompt",
                prompt,
                "--save-llm-output",
                str(llm_output),
                "--output",
                str(scenario_output),
            ]
            print(
                f"GENERATING scenario_{variant_number:03d} attempt {attempt}/{args.max_attempts}",
                flush=True,
            )
            environment = os.environ.copy()
            environment.setdefault("PYTHONIOENCODING", "utf-8")
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )
            attempt_usages = print_captured_output(result.stdout, result.stderr)
            for attempt_usage in attempt_usages:
                scenario_usage = combine_token_usage(scenario_usage, attempt_usage)
                total_usage = combine_token_usage(total_usage, attempt_usage)
                scenario_api_calls += 1
                total_api_calls += 1
                if not attempt_usage.reported:
                    scenario_unreported_calls += 1
                    total_unreported_calls += 1
            if result.returncode != 0:
                print(f"Attempt failed with exit code {result.returncode}", flush=True)
                continue
            try:
                agent_routes, robot_route = load_existing_variant(llm_output, expected_agents)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Rejected invalid output: {exc}", flush=True)
                continue

            signature = scenario_signature(agent_routes, robot_route)
            if signature in signatures:
                print("Rejected duplicate waypoint route combination", flush=True)
                continue

            signatures.add(signature)
            summaries.append(route_summary(variant_number, agent_routes, robot_route))
            generated += 1
            accepted = True
            print(f"GENERATED scenario_{variant_number:03d}", flush=True)
            print(
                f"TOKENS scenario_{variant_number:03d}: "
                f"{format_token_usage(scenario_usage)}; "
                f"{token_usage_suffix(scenario_api_calls, scenario_unreported_calls)}",
                flush=True,
            )
            break

        if not accepted:
            print(
                f"FAILED scenario_{variant_number:03d}: no valid distinct output after "
                f"{args.max_attempts} attempts",
                flush=True,
            )
            print(
                f"TOKENS scenario_{variant_number:03d} attempts: "
                f"{format_token_usage(scenario_usage)}; "
                f"{token_usage_suffix(scenario_api_calls, scenario_unreported_calls)}",
                flush=True,
            )
            print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)
            raise SystemExit(1)

    if args.dry_run:
        print(f"Summary: requested={args.count}, generated=0, existing={existing}")
        return

    if args.skip_previews:
        print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
        print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)
        return

    preview_paths: list[Path] = []
    for variant_number in range(1, args.count + 1):
        variant_dir = scene_output_root / f"scenario_{variant_number:03d}"
        scenario_path = variant_dir / "generated_scenario.yaml"
        preview_path = variant_dir / "scenario_preview.png"
        if preview_path.is_file() and not args.overwrite:
            print(f"EXISTS PREVIEW scenario_{variant_number:03d}", flush=True)
            preview_paths.append(preview_path)
            continue
        print(f"RENDERING scenario_{variant_number:03d}", flush=True)
        render_scenario_preview(renderer, scene_dir, scenario_path, preview_path)
        preview_paths.append(preview_path)

    if args.skip_overview:
        print("SKIPPED OVERVIEW (individual previews were generated)", flush=True)
        print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
        print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)
        return

    overview_path = scene_output_root / "scenarios_overview.png"
    create_overview(preview_paths, overview_path)
    print(f"WROTE OVERVIEW {overview_path}", flush=True)
    print(f"Summary: requested={args.count}, generated={generated}, existing={existing}")
    print_total_token_usage(total_usage, total_api_calls, total_unreported_calls)


if __name__ == "__main__":
    main()
