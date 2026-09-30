#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render individual scenario_preview.png files from existing scenario YAMLs "
            "without using an LLM or API."
        )
    )
    parser.add_argument("--scenes-root", default="new_grscenes")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--only-scene",
        action="append",
        help="Render only this scene ID. May be supplied more than once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate preview images that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes_root = Path(args.scenes_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    renderer = Path(__file__).with_name("topdown_click_to_world.py").resolve()
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"scenes root not found: {scenes_root}")
    if not output_root.is_dir():
        raise FileNotFoundError(f"output root not found: {output_root}")

    requested = set(args.only_scene or [])
    scene_output_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())
    if requested:
        available = {path.name for path in scene_output_dirs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"output scene(s) not found: {', '.join(unknown)}")
        scene_output_dirs = [
            path for path in scene_output_dirs if path.name in requested
        ]

    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    environment.setdefault("MPLBACKEND", "Agg")
    rendered = 0
    existing = 0
    missing_yaml = 0

    for scene_output_dir in scene_output_dirs:
        scene_dir = scenes_root / scene_output_dir.name
        mapping_path = scene_dir / "topdown_mapping.json"
        if not mapping_path.is_file():
            raise FileNotFoundError(
                f"mapping not found for {scene_output_dir.name}: {mapping_path}"
            )
        print(f"=== {scene_output_dir.name} ===", flush=True)
        for variant_dir in sorted(scene_output_dir.glob("scenario_*")):
            if not variant_dir.is_dir():
                continue
            scenario_path = variant_dir / "generated_scenario.yaml"
            preview_path = variant_dir / "scenario_preview.png"
            if not scenario_path.is_file():
                missing_yaml += 1
                print(f"MISSING YAML {variant_dir.name}", flush=True)
                continue
            if preview_path.is_file() and not args.overwrite:
                existing += 1
                print(f"EXISTS {variant_dir.name}", flush=True)
                continue

            command = [
                sys.executable,
                str(renderer),
                "--mapping",
                str(mapping_path),
                "--view",
                "overlay",
                "--scenario",
                str(scenario_path),
                "--save-overlay",
                str(preview_path),
                "--no-show",
            ]
            print(f"RENDERING {variant_dir.name}", flush=True)
            result = subprocess.run(command, check=False, env=environment)
            if result.returncode != 0:
                raise RuntimeError(
                    f"preview renderer failed for {scenario_path} "
                    f"with exit code {result.returncode}"
                )
            rendered += 1

    print(
        f"Preview summary: rendered={rendered}, existing={existing}, "
        f"missing_yaml={missing_yaml}",
        flush=True,
    )


if __name__ == "__main__":
    main()
