#!/usr/bin/env python3
"""
Batch evaluation script for final report experiments.
Runs RL agent and APF baseline across multiple maps and seeds.
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict
import argparse
import time
from tqdm import tqdm

# Experiment configuration - 30 seeds for large sample statistics (n≥30)
SEEDS = [
    20030413,
    20021213,
    7399608,
    42,
    0,
    3407,
    19730422,
    19730328,
    20251120,
    20251121,
    5418,
    7355608,
    7866,
    310101,
    314159,
    2653589793,
    58295651,
    1022065701,
    3068228892,
    328721,
    2123891,
    908263,
    413,
    1213,
    3849203849,
    9834759834,
    1203948576,
    4857203948,
    7192837465,
    5039485720,
]

MAP_CONFIGS = {
    "basic": "configs/env/eval_basic.yaml",
    "medium": "configs/env/eval_medium.yaml",
    "hard": "configs/env/eval_hard.yaml",
}

METHODS = {
    "rl_agent": {"controller": None},  # Use trained model
    "apf": {"controller": "apf"},  # Use APF controller
}

# Evaluation parameters
MODEL_PATH = "/home/gong-zerui/code/ResNav/runs/20251120_115347/final"
STEPS = 600
BASE_OUTPUT_DIR = "runs/final_evaluation"


def run_single_evaluation(
    method: str,
    difficulty: str,
    seed: int,
    env_cfg: str,
    output_dir: Path,
    controller: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a single evaluation rollout.

    Args:
        method: 'rl_agent' or 'apf'
        difficulty: 'basic', 'medium', or 'hard'
        seed: Random seed
        env_cfg: Path to environment config
        output_dir: Directory to save outputs
        controller: Controller type (None for RL, 'apf' for APF)
        verbose: Print detailed output (default: True)

    Returns:
        Dictionary with evaluation metrics
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare output paths
    traj_out = output_dir / "traj.csv"
    metrics_out = output_dir / "metrics.json"
    video_out = output_dir / "eval.mp4"
    snapshot_out = output_dir / "snapshot.png"

    # Build command
    cmd = [
        "python",
        "training/rollout.py",
        "--model",
        MODEL_PATH,
        "--env_cfg",
        env_cfg,
        "--steps",
        str(STEPS),
        "--deterministic",
        "--seed",
        str(seed),
        "--traj_out",
        str(traj_out),
        "--metrics_out",
        str(metrics_out),
        "--record",
        str(video_out),
        "--snapshot_out",
        str(snapshot_out),
    ]

    # Add controller flag for APF
    if controller:
        cmd.extend(["--controller", controller])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {method} | {difficulty} | seed={seed}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")

    # Run evaluation
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            if verbose:
                print(f"ERROR: Evaluation failed!")
                print(f"STDERR: {result.stderr}")
            return {
                "method": method,
                "difficulty": difficulty,
                "seed": seed,
                "success": False,
                "error": result.stderr,
            }

        # Load metrics
        with open(metrics_out, "r") as f:
            metrics = json.load(f)

        # Add metadata
        metrics["method"] = method
        metrics["difficulty"] = difficulty
        metrics["seed"] = seed

        if verbose:
            print(f"✓ Success: {metrics.get('success', False)}")

            # Safe formatting for optional metrics (use correct field names from rollout.py)
            time_val = metrics.get("time_elapsed_s")
            if time_val is not None and isinstance(time_val, (int, float)):
                print(f"  Time: {time_val:.2f}s")
            else:
                print(f"  Time: N/A")

            path_val = metrics.get("path_length_m")
            if path_val is not None and isinstance(path_val, (int, float)):
                print(f"  Path: {path_val:.2f}m")
            else:
                print(f"  Path: N/A")

            smooth_val = metrics.get("smoothness_score")
            if smooth_val is not None and isinstance(smooth_val, (int, float)):
                print(f"  Smoothness: {smooth_val:.4f}")
            else:
                print(f"  Smoothness: N/A")

        # Normalize field names for consistency in CSV output
        metrics["time_s"] = metrics.get("time_elapsed_s")
        metrics["path_smoothness"] = metrics.get("smoothness_score")

        return metrics

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"ERROR: Evaluation timed out after 5 minutes")
        return {
            "method": method,
            "difficulty": difficulty,
            "seed": seed,
            "success": False,
            "error": "Timeout",
        }
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return {
            "method": method,
            "difficulty": difficulty,
            "seed": seed,
            "success": False,
            "error": str(e),
        }


def run_batch_evaluation(
    methods: List[str] = None,
    difficulties: List[str] = None,
    seeds: List[int] = None,
    skip_existing: bool = True,
):
    """
    Run batch evaluation across all configurations.

    Args:
        methods: List of methods to evaluate (default: all)
        difficulties: List of difficulties to test (default: all)
        seeds: List of seeds to use (default: all)
        skip_existing: Skip experiments that already have results (default: True)
    """
    # Use defaults if not specified
    if methods is None:
        methods = list(METHODS.keys())
    if difficulties is None:
        difficulties = list(MAP_CONFIGS.keys())
    if seeds is None:
        seeds = SEEDS

    # Build task list
    tasks = []
    for method in methods:
        for difficulty in difficulties:
            for seed in seeds:
                output_dir = (
                    Path(BASE_OUTPUT_DIR) / method / difficulty / f"seed_{seed}"
                )
                metrics_file = output_dir / "metrics.json"

                # Check if already exists
                if skip_existing and metrics_file.exists():
                    continue

                tasks.append(
                    {
                        "method": method,
                        "difficulty": difficulty,
                        "seed": seed,
                        "env_cfg": MAP_CONFIGS[difficulty],
                        "output_dir": output_dir,
                        "controller": METHODS[method]["controller"],
                    }
                )

    total = len(tasks)
    total_planned = len(methods) * len(difficulties) * len(seeds)
    skipped = total_planned - total

    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION")
    print(f"{'='*80}")
    print(f"Methods: {methods}")
    print(f"Difficulties: {difficulties}")
    print(f"Seeds: {len(seeds)} seeds")
    print(f"Total planned: {total_planned}")
    if skipped > 0:
        print(f"Already completed: {skipped} (skipping)")
    print(f"To run: {total}")
    print(f"{'='*80}\n")

    if total == 0:
        print("✓ All experiments already completed!")
        return pd.read_csv(Path(BASE_OUTPUT_DIR) / "results" / "detailed_results.csv")

    all_results = []
    start_time = time.time()

    # Progress bar
    pbar = tqdm(
        tasks,
        desc="Running experiments",
        unit="rollout",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i, task in enumerate(pbar):
        # Update description with current task
        desc = f"{task['method']:9s} | {task['difficulty']:6s} | seed_{task['seed']}"
        pbar.set_description(desc)

        # Run evaluation (suppress detailed output)
        result = run_single_evaluation(
            method=task["method"],
            difficulty=task["difficulty"],
            seed=task["seed"],
            env_cfg=task["env_cfg"],
            output_dir=task["output_dir"],
            controller=task["controller"],
            verbose=False,  # Suppress detailed output
        )

        all_results.append(result)

        # Update postfix with success info
        success_count = sum(1 for r in all_results if r.get("success", False))
        success_rate = (success_count / len(all_results)) * 100
        pbar.set_postfix({"success_rate": f"{success_rate:.1f}%"})

    pbar.close()

    # Calculate statistics
    elapsed = time.time() - start_time
    avg_time = elapsed / total if total > 0 else 0

    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE!")
    print(f"{'='*80}")
    print(f"Completed: {total} experiments")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average time per rollout: {avg_time:.1f}s")
    print(f"Overall success rate: {success_rate:.1f}%")
    print(f"{'='*80}\n")

    # Save all results
    results_dir = Path(BASE_OUTPUT_DIR) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    results_csv = results_dir / "detailed_results.csv"

    # If we skipped some, merge with existing results
    if skip_existing and results_csv.exists():
        existing_df = pd.read_csv(results_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["method", "difficulty", "seed"], keep="last")

    df.to_csv(results_csv, index=False)
    print(f"All results saved to: {results_csv}")

    # Generate summary statistics
    generate_summary(df, results_dir)

    return df


def generate_summary(df: pd.DataFrame, output_dir: Path):
    """
    Generate summary statistics from detailed results.

    Args:
        df: DataFrame with all detailed results
        output_dir: Directory to save summary
    """
    print(f"\nGenerating summary statistics...")

    # Group by method and difficulty
    summary_data = []

    for method in df["method"].unique():
        for difficulty in df["difficulty"].unique():
            subset = df[(df["method"] == method) & (df["difficulty"] == difficulty)]

            if len(subset) == 0:
                continue

            # Calculate statistics
            success_count = subset["success"].sum()
            total_count = len(subset)
            success_rate = (success_count / total_count) * 100

            # Only calculate stats for successful runs
            successful = subset[subset["success"] == True]

            row = {
                "method": method,
                "difficulty": difficulty,
                "total_runs": total_count,
                "successes": success_count,
                "success_rate_%": success_rate,
                "mean_time_s": (
                    successful["time_s"].mean() if len(successful) > 0 else None
                ),
                "std_time_s": (
                    successful["time_s"].std() if len(successful) > 0 else None
                ),
                "mean_path_length_m": (
                    successful["path_length_m"].mean() if len(successful) > 0 else None
                ),
                "std_path_length_m": (
                    successful["path_length_m"].std() if len(successful) > 0 else None
                ),
                "mean_smoothness": (
                    successful["path_smoothness"].mean()
                    if len(successful) > 0
                    else None
                ),
                "std_smoothness": (
                    successful["path_smoothness"].std() if len(successful) > 0 else None
                ),
            }

            summary_data.append(row)

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Summary saved to: {summary_csv}")
    print(f"\nSummary Statistics:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for final report experiments"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rl_agent", "apf"],
        default=None,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=["basic", "medium", "hard"],
        default=None,
        help="Difficulty levels to test (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to use (default: predefined list)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with only 2 seeds on basic difficulty",
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE ⚡")
        methods = args.methods or ["rl_agent", "apf"]
        difficulties = ["basic"]
        seeds = [20030413, 42]
    else:
        methods = args.methods
        difficulties = args.difficulties
        seeds = args.seeds

    # Run batch evaluation
    df = run_batch_evaluation(methods=methods, difficulties=difficulties, seeds=seeds)

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"Total experiments: {len(df)}")
    print(f"Successful: {df['success'].sum()}")
    print(f"Failed: {(~df['success']).sum()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
