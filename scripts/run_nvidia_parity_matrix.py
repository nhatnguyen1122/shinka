#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_convergence import export_convergence


BENCHMARKS = {
    "circle_packing": {"alias": "cp", "config": "examples/openevolve_parity/circle_packing_nvidia.yaml"},
    "circle_packing_n32": {
        "alias": "cp32",
        "config": "examples/openevolve_parity/circle_packing_n32_nvidia.yaml",
    },
    "circle_packing_rect": {
        "alias": "cprect",
        "config": "examples/openevolve_parity/circle_packing_rect_nvidia.yaml",
    },
    "erdos_min_overlap": {
        "alias": "erdos",
        "config": "examples/openevolve_parity/erdos_min_overlap_nvidia.yaml",
    },
    "first_autocorr_ineq": {
        "alias": "auto1",
        "config": "examples/openevolve_parity/first_autocorr_ineq_nvidia.yaml",
    },
    "heilbronn_convex/13": {
        "alias": "hconv13",
        "config": "examples/openevolve_parity/hconv13_nvidia.yaml",
    },
    "heilbronn_convex/14": {
        "alias": "hconv14",
        "config": "examples/openevolve_parity/hconv14_nvidia.yaml",
    },
    "heilbronn_triangle": {
        "alias": "heil",
        "config": "examples/openevolve_parity/heilbronn_triangle_nvidia.yaml",
    },
    "hexagon_packing/11": {
        "alias": "hex11",
        "config": "examples/openevolve_parity/hex11_nvidia.yaml",
    },
    "hexagon_packing/12": {
        "alias": "hex12",
        "config": "examples/openevolve_parity/hex12_nvidia.yaml",
    },
    "matmul": {
        "alias": "matmul",
        "config": "examples/openevolve_parity/matmul_nvidia.yaml",
    },
    "minimizing_max_min_dist/2": {
        "alias": "mmd2",
        "config": "examples/openevolve_parity/mmd2_nvidia.yaml",
    },
    "minimizing_max_min_dist/3": {
        "alias": "mmd3",
        "config": "examples/openevolve_parity/mmd3_nvidia.yaml",
    },
    "second_autocorr_ineq": {
        "alias": "auto2",
        "config": "examples/openevolve_parity/second_autocorr_ineq_nvidia.yaml",
    },
    "sums_diffs_finite_sets": {
        "alias": "sumsdiffs",
        "config": "examples/openevolve_parity/sums_diffs_finite_sets_nvidia.yaml",
    },
    "third_autocorr_ineq": {
        "alias": "auto3",
        "config": "examples/openevolve_parity/third_autocorr_ineq_nvidia.yaml",
    },
    "uncertainty_ineq": {
        "alias": "uncert",
        "config": "examples/openevolve_parity/uncertainty_ineq_nvidia.yaml",
    },
    "function_minimization": {
        "alias": "func",
        "config": "examples/openevolve_parity/function_minimization_nvidia.yaml",
    },
    "k_module_problem": {"alias": "kmod", "config": "examples/openevolve_parity/k_module_problem_nvidia.yaml"},
    "signal_processing": {"alias": "sp", "config": "examples/openevolve_parity/signal_processing_nvidia.yaml"},
}


def _parse_benchmarks(value: str) -> list[str]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    aliases = {cfg["alias"]: name for name, cfg in BENCHMARKS.items()}
    valid = set(BENCHMARKS) | set(aliases)
    unknown = [item for item in raw if item not in valid]
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}")
    return [aliases.get(item, item) for item in raw]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _best_from_convergence(path: Path) -> tuple[float | None, dict[str, Any]]:
    best_score = None
    best_row: dict[str, Any] = {}
    if not path.exists():
        return None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if best_score is None or float(score) > best_score:
                best_score = float(score)
                best_row = row
    return best_score, best_row


def run_one(
    benchmark: str,
    run_idx: int,
    output_root: Path,
    generations: int | None,
    resume: bool,
) -> dict[str, Any]:
    cfg = BENCHMARKS[benchmark]
    run_dir = output_root / benchmark / "shinka" / f"run_{run_idx:02d}"
    convergence_path = run_dir / "convergence.jsonl"
    if resume and convergence_path.exists():
        best_score, best_row = _best_from_convergence(convergence_path)
        return {
            "benchmark": benchmark,
            "strategy": "shinka",
            "run": run_idx,
            "status": "skipped",
            "output_dir": str(run_dir),
            "primary_score": best_score,
            "metrics": best_row.get("metrics", {}),
        }

    command = [
        sys.executable,
        "examples/openevolve_parity/run_parity.py",
        "--config_path",
        cfg["config"],
        "--results_dir",
        str(run_dir),
    ]
    if generations is not None:
        command.extend(["--generations", str(generations)])

    start = time.time()
    proc = subprocess.run(command, text=True)
    elapsed = time.time() - start

    export_convergence(run_dir, convergence_path)
    best_score, best_row = _best_from_convergence(convergence_path)
    return {
        "benchmark": benchmark,
        "strategy": "shinka",
        "run": run_idx,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "output_dir": str(run_dir),
        "convergence_path": str(convergence_path),
        "primary_score": best_score,
        "metrics": best_row.get("metrics", {}),
        "elapsed_seconds": elapsed,
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve NVIDIA parity matrix")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--benchmarks", default="circle_packing,function_minimization,k_module_problem,signal_processing")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output-root", default="results/nvidia_parity_matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")

    benchmarks = _parse_benchmarks(args.benchmarks)
    output_root = Path(args.output_root)
    results_path = output_root / "results.jsonl"

    for benchmark in benchmarks:
        for run_idx in range(1, args.runs + 1):
            print(f"[{benchmark} / shinka / run {run_idx}]")
            record = run_one(
                benchmark=benchmark,
                run_idx=run_idx,
                output_root=output_root,
                generations=args.generations,
                resume=args.resume,
            )
            _append_jsonl(results_path, record)
            if args.fail_fast and record["status"] == "failed":
                raise SystemExit(record.get("returncode") or 1)

    print(f"Wrote matrix results to {results_path}")


if __name__ == "__main__":
    main()
