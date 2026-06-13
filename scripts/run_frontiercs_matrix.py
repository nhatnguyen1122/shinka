#!/usr/bin/env python3
"""Run Frontier-CS problems with ShinkaEvolve."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_convergence import export_convergence  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE_ROOT = WORKSPACE / "AdaEvolveMCTS_outer_level_mcts" / "benchmarks" / "frontier-cs-eval"
FRONTIER_DIR = SOURCE_ROOT / "Frontier-CS"


def available_problems() -> list[str]:
    problems = FRONTIER_DIR / "algorithmic" / "problems"
    if not problems.exists():
        return []
    return sorted((p.name for p in problems.iterdir() if p.is_dir() and p.name.isdigit()), key=int)


def parse_csv(value: str) -> list[str]:
    if value == "all":
        problems = available_problems()
        if not problems:
            raise ValueError("No Frontier-CS problems found. Clone Frontier-CS first.")
        return problems
    return [x.strip() for x in value.split(",") if x.strip()]


def best_from_convergence(path: Path) -> tuple[float | None, dict[str, Any]]:
    best_score = None
    best_row: dict[str, Any] = {}
    if not path.exists():
        return None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score")
        if isinstance(score, (int, float)) and (best_score is None or score > best_score):
            best_score = float(score)
            best_row = row
    return best_score, best_row


def problem_prompt(problem_id: str) -> str:
    statement = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "statement.txt"
    config = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "config.yaml"
    statement_text = statement.read_text(encoding="utf-8") if statement.exists() else f"Frontier-CS problem {problem_id}"
    config_text = config.read_text(encoding="utf-8") if config.exists() else ""
    return f"""You are an expert competitive programmer specializing in algorithmic optimization.

FRONTIER-CS PROBLEM ID: {problem_id}

PROBLEM STATEMENT:
{statement_text}

CONSTRAINTS AND JUDGE CONFIG:
{config_text}

OBJECTIVE: Maximize the score returned by the Frontier-CS judge. Higher is better.
Your solution must be valid C++ code with main(), reading from stdin and writing to stdout.
Return complete C++ code only.
"""


def prepare_benchmark(problem_id: str, output_root: Path) -> Path:
    dest = output_root / "_benchmarks" / f"problem_{problem_id}"
    dest.mkdir(parents=True, exist_ok=True)
    code = (SOURCE_ROOT / "initial_program.cpp").read_text(encoding="utf-8")
    if "EVOLVE-BLOCK-START" not in code:
        code = f"// EVOLVE-BLOCK-START\n{code.rstrip()}\n// EVOLVE-BLOCK-END\n"
    (dest / "initial_program.py").write_text(code, encoding="utf-8")
    (dest / "evaluator.py").write_text(
        f"""
from __future__ import annotations
import importlib.util
from pathlib import Path

_path = Path({str(SOURCE_ROOT / "evaluator.py")!r})
_spec = importlib.util.spec_from_file_location("frontiercs_source_evaluator", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator from {{_path}}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
evaluate = _module.evaluate
""".lstrip(),
        encoding="utf-8",
    )
    return dest


def prepare_config(problem_id: str, bench_dir: Path, output_root: Path, generations: int) -> Path:
    cfg = {
        "benchmark_dir": str(bench_dir),
        "eval_time": "00:10:00",
        "max_evaluation_jobs": 1,
        "max_proposal_jobs": 1,
        "max_db_workers": 1,
        "db_config": {
            "num_islands": 4,
            "archive_size": 30,
            "elite_selection_ratio": 0.2,
            "num_archive_inspirations": 2,
            "num_top_k_inspirations": 1,
            "migration_interval": 10,
            "migration_rate": 0.1,
            "island_elitism": True,
            "enforce_island_separation": True,
            "parent_selection_strategy": "weighted",
            "archive_selection_strategy": "crowding",
            "island_selection_strategy": "equal",
            "archive_criteria": {"combined_score": 1.0, "loc": -0.1},
        },
        "evo_config": {
            "task_sys_msg": problem_prompt(problem_id),
            "patch_types": ["full", "diff", "cross"],
            "patch_type_probs": [0.4, 0.5, 0.1],
            "num_generations": generations,
            "max_patch_resamples": 3,
            "max_patch_attempts": 1,
            "job_type": "local",
            "language": "cpp",
            "llm_models": ["local/openai/gpt-oss-120b@https://integrate.api.nvidia.com/v1?api_key_env=NVIDIA_API_KEY"],
            "llm_dynamic_selection": "fixed",
            "llm_kwargs": {"temperatures": [0.2], "max_tokens": 4096, "reasoning_efforts": ["disabled"]},
            "meta_rec_interval": None,
            "embedding_model": None,
            "results_dir": str(output_root / "_default_results" / f"problem_{problem_id}"),
            "max_novelty_attempts": 1,
            "max_api_costs": None,
            "evolve_prompts": False,
        },
    }
    cfg_dir = output_root / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"problem_{problem_id}_shinka.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="0", help="Comma list of problem IDs, or 'all'.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--output-root", default="results/frontiercs_shinka")
    parser.add_argument("--judge-urls", default=os.environ.get("JUDGE_URLS", "http://localhost:8081"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    problems = parse_csv(args.problems)
    if not args.dry_run and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for problem_id in problems:
        bench_dir = prepare_benchmark(problem_id, output_root)
        cfg = prepare_config(problem_id, bench_dir, output_root, args.generations)
        for run_idx in range(1, args.runs + 1):
            run_dir = output_root / f"problem_{problem_id}" / "shinka" / f"run_{run_idx:02d}"
            convergence = run_dir / "convergence.jsonl"
            if args.resume and convergence.exists():
                score, row = best_from_convergence(convergence)
                rows.append({"problem": problem_id, "strategy": "shinka", "run": run_idx, "status": "skipped", "primary_score": score, "metrics": row.get("metrics", {})})
                continue
            cmd = [
                sys.executable,
                "examples/openevolve_parity/run_parity.py",
                "--config_path",
                str(cfg),
                "--results_dir",
                str(run_dir),
                "--generations",
                str(args.generations),
            ]
            print("+ " + " ".join(cmd))
            if args.dry_run:
                continue
            env = os.environ.copy()
            env["FRONTIER_CS_PROBLEM"] = str(problem_id)
            env["JUDGE_URLS"] = args.judge_urls
            start = time.time()
            proc = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
            export_convergence(run_dir, convergence)
            score, best = best_from_convergence(convergence)
            record = {
                "problem": problem_id,
                "strategy": "shinka",
                "run": run_idx,
                "status": "ok" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "output_dir": str(run_dir),
                "elapsed_seconds": round(time.time() - start, 3),
                "primary_score": score,
                "metrics": best.get("metrics", {}),
            }
            rows.append(record)
            with (output_root / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            print(f"[problem {problem_id} run {run_idx}] {record['status']} score={score}")
            if args.fail_fast and proc.returncode != 0:
                return proc.returncode or 1
    if rows:
        keys = sorted({k for row in rows for k in row if k != "metrics"})
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "metrics"} for row in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
