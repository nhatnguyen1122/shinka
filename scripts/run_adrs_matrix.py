#!/usr/bin/env python3
"""Run ADRS benchmarks with ShinkaEvolve."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
SOURCE_ROOT = WORKSPACE / "AdaEvolveMCTS_outer_level_mcts" / "benchmarks" / "ADRS"
BENCHMARKS = {
    "cloudcast": "cloudcast",
    "eplb": "eplb",
    "llm_sql": "llm_sql",
    "prism": "prism",
    "txn_scheduling": "txn_scheduling",
}
ALIASES = {"txn": "txn_scheduling", "sql": "llm_sql"}


def parse_csv(value: str) -> list[str]:
    items = [ALIASES.get(x.strip(), x.strip()) for x in value.split(",") if x.strip()]
    unknown = sorted(set(items) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown ADRS benchmark(s): {', '.join(unknown)}")
    return items


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def prepare_benchmark(benchmark: str, output_root: Path) -> Path:
    dest = output_root / "_benchmarks" / benchmark
    if dest.exists():
        return dest
    shutil.copytree(SOURCE_ROOT / benchmark, dest)
    wrapper = dest / "evaluator.py"
    wrapper.write_text(
        """
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_evaluator_dir = Path(__file__).resolve().parent / "evaluator"
if str(_evaluator_dir) not in sys.path:
    sys.path.insert(0, str(_evaluator_dir))
_path = _evaluator_dir / "evaluator.py"
_spec = importlib.util.spec_from_file_location("adrs_nested_evaluator", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load evaluator from {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
evaluate = _module.evaluate
""".lstrip(),
        encoding="utf-8",
    )
    return dest


def prepare_config(benchmark: str, bench_dir: Path, output_root: Path, generations: int) -> Path:
    with (SOURCE_ROOT / benchmark / "config.yaml").open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    cfg = {
        "benchmark_dir": str(bench_dir),
        "eval_time": "00:10:00",
        "max_evaluation_jobs": 2,
        "max_proposal_jobs": 2,
        "max_db_workers": 2,
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
            "task_sys_msg": (base.get("prompt") or {}).get("system_message", ""),
            "patch_types": ["full", "diff", "cross"],
            "patch_type_probs": [0.4, 0.5, 0.1],
            "num_generations": generations,
            "max_patch_resamples": 3,
            "max_patch_attempts": 1,
            "job_type": "local",
            "language": "python",
            "llm_models": [
                "local/openai/gpt-oss-120b@https://integrate.api.nvidia.com/v1?api_key_env=NVIDIA_API_KEY"
            ],
            "llm_dynamic_selection": "fixed",
            "llm_kwargs": {"temperatures": [0.2], "max_tokens": 4096, "reasoning_efforts": ["disabled"]},
            "meta_rec_interval": None,
            "embedding_model": None,
            "results_dir": str(output_root / "_default_results" / benchmark),
            "max_novelty_attempts": 1,
            "max_api_costs": None,
            "evolve_prompts": False,
        },
    }
    cfg_dir = output_root / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{benchmark}_shinka.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--output-root", default="results/adrs_shinka")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    benchmarks = parse_csv(args.benchmarks)
    if not args.dry_run and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        bench_dir = prepare_benchmark(benchmark, output_root)
        cfg = prepare_config(benchmark, bench_dir, output_root, args.generations)
        for run_idx in range(1, args.runs + 1):
            run_dir = output_root / benchmark / "shinka" / f"run_{run_idx:02d}"
            convergence = run_dir / "convergence.jsonl"
            if args.resume and convergence.exists():
                score, row = best_from_convergence(convergence)
                rows.append({"benchmark": benchmark, "run": run_idx, "status": "skipped", "primary_score": score, "metrics": row.get("metrics", {})})
                continue
            cmd = [
                sys.executable, "examples/openevolve_parity/run_parity.py",
                "--config_path", str(cfg),
                "--results_dir", str(run_dir),
                "--generations", str(args.generations),
            ]
            print("+ " + " ".join(cmd))
            if args.dry_run:
                continue
            start = time.time()
            proc = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=False)
            export_convergence(run_dir, convergence)
            score, best = best_from_convergence(convergence)
            record = {
                "benchmark": benchmark,
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
            print(f"[{benchmark} run {run_idx}] {record['status']} score={score}")
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
