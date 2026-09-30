#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from generate_scenario_from_llm import TokenUsage, combine_token_usage, format_token_usage


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    size: str
    waypoint_count: int
    scenario_count: int
    min_agents: int
    max_agents: int


# Fixed 10k allocation based on the current dense waypoint set.
# Small: 1-5 agents; medium: 1-8 agents; large: 1-10 agents.
SCENE_SPECS = (
    SceneSpec("MV4AFHQKTKJZ2AABAAAAADQ8_usd", "medium", 181, 320, 1, 8),
    SceneSpec("MV4AFHQKTKJZ2AABAAAAADY8_usd", "medium", 274, 348, 1, 8),
    SceneSpec("MV4AFHQKTKJZ2AABAAAAAEA8_usd", "medium", 261, 345, 1, 8),
    SceneSpec("MV4AFHQKTKJZ2AABAAAAAEI8_usd", "medium", 172, 317, 1, 8),
    SceneSpec("MV5M25QKTKJZ2AABAAAAAAA8_usd", "medium", 240, 339, 1, 8),
    SceneSpec("MV5M25QKTKJZ2AABAAAAAAI8_usd", "medium", 208, 329, 1, 8),
    SceneSpec("MV5M25QKTKJZ2AABAAAAAAQ8_usd", "medium", 233, 337, 1, 8),
    SceneSpec("MV5M25QKTKJZ2AABAAAAAAY8_usd", "medium", 208, 329, 1, 8),
    SceneSpec("MV5M25QKTKJZ2AABAAAAAEI8_usd", "medium", 177, 319, 1, 8),
    SceneSpec("MV7J6NIKTKJZ2AABAAAAAAA8_usd", "medium", 191, 324, 1, 8),
    SceneSpec("MV7J6NIKTKJZ2AABAAAAAAI8_usd", "medium", 177, 319, 1, 8),
    SceneSpec("MVJWVGYKTLDAYAABAAAAAAQ8_usd", "small", 142, 307, 1, 5),
    SceneSpec("MVSGSAIKTKJ66AABAAAAADY8_usd", "small", 97, 288, 1, 5),
    SceneSpec("MVSGSAIKTKJ66AABAAAAAEA8_usd", "small", 159, 313, 1, 5),
    SceneSpec("MVSYCXYKTKJ66AABAAAAAAA8_usd", "small", 100, 289, 1, 5),
    SceneSpec("MVSYCXYKTKJ66AABAAAAACY8_usd", "medium", 192, 324, 1, 8),
    SceneSpec("MVSYCXYKTKJ66AABAAAAADA8_usd", "small", 86, 283, 1, 5),
    SceneSpec("MVSYCXYKTKJ66AABAAAAADI8_usd", "small", 156, 312, 1, 5),
    SceneSpec("MWF4WLIKTIFZIAABAAAAABY8_usd", "large", 694, 436, 1, 10),
    SceneSpec("MWF4WLIKTIFZIAABAAAAACA8_usd", "large", 468, 394, 1, 10),
    SceneSpec("MWF4WLIKTIFZIAABAAAAACI8_usd", "medium", 192, 324, 1, 8),
    SceneSpec("MWF4WLIKTIFZIAABAAAAACQ8_usd", "small", 167, 316, 1, 5),
    SceneSpec("MWF4WLIKTIFZIAABAAAAACY8_usd", "large", 760, 447, 1, 10),
    SceneSpec("MWF4WLIKTIFZIAABAAAAADA8_usd", "large", 520, 404, 1, 10),
    SceneSpec("MWF4WLIKTIFZIAABAAAAADI8_usd", "medium", 219, 332, 1, 8),
    SceneSpec("MWF4WLIKTIFZIAABAAAAADQ8_usd", "small", 94, 287, 1, 5),
    SceneSpec("MWF4WLIKTIFZIAABAAAAADY8_usd", "medium", 347, 367, 1, 8),
    SceneSpec("MWF4WLIKTIFZIAABAAAAAEA8_usd", "large", 455, 391, 1, 10),
    SceneSpec("MWF4WLIKTIFZIAABAAAAAEI8_usd", "small", 83, 282, 1, 5),
    SceneSpec("MWHLEPQKTIFZIAABAAAAAAA8_usd", "small", 75, 278, 1, 5),
)

EXPECTED_TOTAL = 10_000
TOKEN_LINE = re.compile(
    r"^TOKENS (?:scenario_\d+(?: attempts)?|batch_\d+_\d+): "
    r"input=(\d+), output=(\d+), reasoning=(\d+), cached=(\d+), total=(\d+); "
    r"api_calls=(\d+)(?:, usage_unreported_calls=(\d+))?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed 10,000-scenario dataset across all 30 scenes."
    )
    parser.add_argument("--scenes-root", default="new_grscenes")
    parser.add_argument("--output-root", default="new_outputs")
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--history-window", type=int, default=12)
    parser.add_argument(
        "--only-scene",
        action="append",
        help="Generate only this fixed scene ID. May be supplied more than once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-previews",
        action="store_true",
        help="Render individual previews after generation. Disabled by default.",
    )
    parser.add_argument(
        "--render-overview",
        action="store_true",
        help="Also combine all previews for each scene into one overview image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed plan without calling the LLM or writing output files.",
    )
    return parser.parse_args()


def validate_fixed_specs() -> None:
    if len(SCENE_SPECS) != 30:
        raise RuntimeError(f"fixed plan must contain 30 scenes, got {len(SCENE_SPECS)}")
    if len({spec.scene_id for spec in SCENE_SPECS}) != len(SCENE_SPECS):
        raise RuntimeError("fixed plan contains duplicate scene IDs")
    actual_total = sum(spec.scenario_count for spec in SCENE_SPECS)
    if actual_total != EXPECTED_TOTAL:
        raise RuntimeError(
            f"fixed plan must contain {EXPECTED_TOTAL} scenarios, got {actual_total}"
        )


def print_plan(specs: list[SceneSpec]) -> None:
    print("Fixed scenario allocation:")
    for selected_scene_index, spec in enumerate(specs, start=1):
        print(
            f"  {spec.scene_id}: size={spec.size}, waypoints={spec.waypoint_count}, "
            f"scenarios={spec.scenario_count}, agents={spec.min_agents}-{spec.max_agents}"
        )
    print(f"Selected total: {sum(spec.scenario_count for spec in specs)} scenarios")


def print_dataset_tokens(
    usage: TokenUsage,
    api_calls: int,
    unreported_calls: int,
) -> None:
    suffix = f"api_calls={api_calls}"
    if unreported_calls:
        suffix += f", usage_unreported_calls={unreported_calls}"
    print(f"TOTAL TOKENS ALL SCENES: {format_token_usage(usage)}; {suffix}", flush=True)


def main() -> None:
    args = parse_args()
    validate_fixed_specs()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.history_window < 0:
        raise ValueError("--history-window cannot be negative")

    specs = list(SCENE_SPECS)
    if args.only_scene:
        requested = set(args.only_scene)
        known = {spec.scene_id for spec in SCENE_SPECS}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown fixed scene ID(s): {', '.join(unknown)}")
        specs = [spec for spec in specs if spec.scene_id in requested]

    print_plan(specs)
    if args.dry_run:
        return

    scenes_root = Path(args.scenes_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    llm_config = Path(args.llm_config).expanduser().resolve()
    variant_generator = Path(__file__).with_name("generate_scenario_variants.py").resolve()
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"scenes root not found: {scenes_root}")
    if not llm_config.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")

    total_usage = TokenUsage()
    total_api_calls = 0
    total_unreported_calls = 0
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")

    fixed_scene_indexes = {spec.scene_id: index for index, spec in enumerate(SCENE_SPECS)}
    for spec in specs:
        scene_dir = scenes_root / spec.scene_id
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"fixed scene directory not found: {scene_dir}")

        print(
            f"\n=== SCENE {selected_scene_index}/{len(specs)} | "
            f"{spec.scene_id} ({spec.size}) | {spec.scenario_count} scenarios | "
            f"agents {spec.min_agents}-{spec.max_agents} ===",
            flush=True,
        )
        command = [
            sys.executable,
            str(variant_generator),
            "--scene-dir",
            str(scene_dir),
            "--output-root",
            str(output_root),
            "--llm-config",
            str(llm_config),
            "--prompt",
            args.prompt,
            "--count",
            str(spec.scenario_count),
            "--batch-size",
            "10",
            "--min-pedestrians",
            str(spec.min_agents),
            "--max-pedestrians",
            str(spec.max_agents),
            "--seed",
            str(args.seed + fixed_scene_indexes[spec.scene_id]),
            "--history-window",
            str(args.history_window),
            "--max-attempts",
            str(args.max_attempts),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if not args.render_previews and not args.render_overview:
            command.append("--skip-previews")
        elif not args.render_overview:
            command.append("--skip-overview")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            match = TOKEN_LINE.match(line.strip())
            if match:
                values = [int(value or 0) for value in match.groups()]
                total_usage = combine_token_usage(
                    total_usage,
                    TokenUsage(
                        input_tokens=values[0],
                        output_tokens=values[1],
                        reasoning_tokens=values[2],
                        cached_tokens=values[3],
                        total_tokens=values[4],
                        reported=True,
                    ),
                )
                total_api_calls += values[5]
                total_unreported_calls += values[6]
        return_code = process.wait()
        if return_code != 0:
            print_dataset_tokens(total_usage, total_api_calls, total_unreported_calls)
            raise SystemExit(
                f"scene generation failed for {spec.scene_id} with exit code {return_code}"
            )
        print(
            f"COMPLETED SCENE {selected_scene_index}/{len(specs)}: {spec.scene_id}",
            flush=True,
        )

    print_dataset_tokens(total_usage, total_api_calls, total_unreported_calls)


if __name__ == "__main__":
    main()
