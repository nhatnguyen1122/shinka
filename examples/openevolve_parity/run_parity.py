#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


ROOT = Path(__file__).resolve().parent


def main(
    config_path: str,
    generations: int | None = None,
    results_dir: str | None = None,
) -> None:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    benchmark_dir = (config_file.parent / config["benchmark_dir"]).resolve()
    evaluator_path = benchmark_dir / "evaluator.py"
    initial_path = benchmark_dir / "initial_program.py"

    evo_values = {**config["evo_config"], "init_program_path": str(initial_path)}
    if generations is not None:
        evo_values["num_generations"] = generations
    if results_dir is not None:
        evo_values["results_dir"] = results_dir
    evo_config = EvolutionConfig(**evo_values)
    db_config = DatabaseConfig(**config["db_config"])
    job_config = LocalJobConfig(
        eval_program_path=str(ROOT / "evaluate_adapter.py"),
        time=config.get("eval_time", "00:05:00"),
        extra_cmd_args={"evaluator_path": str(evaluator_path)},
    )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs", 2),
        max_proposal_jobs=config.get("max_proposal_jobs", 2),
        max_db_workers=config.get("max_db_workers", 2),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()
    main(args.config_path, generations=args.generations, results_dir=args.results_dir)
