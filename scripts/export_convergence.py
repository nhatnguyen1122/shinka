#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SCORE_KEYS = ("combined_score", "overall_score", "composite_score", "score", "fitness", "accuracy")


def _json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _score_from_metrics(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in SCORE_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    numeric = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric:
        return "numeric_average", float(sum(numeric) / len(numeric))
    return None, None


def _db_path(run_dir: Path) -> Path:
    if run_dir.suffix in {".sqlite", ".db"}:
        return run_dir
    return run_dir / "programs.sqlite"


def export_convergence(run_dir: Path, output_path: Path, include_island_copies: bool = False) -> int:
    db_path = _db_path(run_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"Program database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(programs)")}
        wanted = [
            col
            for col in (
                "id",
                "generation",
                "timestamp",
                "parent_id",
                "combined_score",
                "correct",
                "metrics",
                "public_metrics",
                "private_metrics",
                "metadata",
            )
            if col in columns
        ]
        rows = [dict(row) for row in conn.execute(f"SELECT {', '.join(wanted)} FROM programs")]
    finally:
        conn.close()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        metrics = {}
        metrics.update(_json_dict(row.get("metadata")))
        metrics.update(_json_dict(row.get("public_metrics")))
        metrics.update(_json_dict(row.get("private_metrics")))
        metrics.update(_json_dict(row.get("metrics")))
        if row.get("combined_score") is not None:
            metrics["combined_score"] = row.get("combined_score")
        if not include_island_copies and metrics.get("_is_island_copy"):
            continue

        primary_metric, score = _score_from_metrics(metrics)
        if score is None:
            continue
        normalized.append(
            {
                "iteration": int(row.get("generation") or 0),
                "event": "program",
                "program_id": row.get("id"),
                "parent_id": row.get("parent_id"),
                "score": score,
                "primary_metric": primary_metric,
                "metrics": metrics,
                "correct": bool(row.get("correct")),
                "timestamp": row.get("timestamp"),
                "source": str(db_path),
            }
        )

    normalized.sort(key=lambda row: (int(row.get("iteration") or 0), row.get("timestamp") or 0, row.get("program_id") or ""))
    best_score: float | None = None
    for row in normalized:
        score = row.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            best_score = float(score) if best_score is None else max(best_score, float(score))
        row["best_score"] = best_score

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, default=str) + "\n")
    return len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ShinkaEvolve run convergence.jsonl")
    parser.add_argument("run_dir", help="ShinkaEvolve result directory or programs.sqlite path")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL path")
    parser.add_argument(
        "--include-island-copies",
        action="store_true",
        help="Include synthetic initial-program copies created for each island",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else (run_dir if run_dir.suffix else run_dir / "convergence.jsonl")
    if output == run_dir:
        output = run_dir.with_name("convergence.jsonl")
    count = export_convergence(run_dir, output, include_island_copies=args.include_island_copies)
    print(f"Wrote {count} convergence rows to {output}")


if __name__ == "__main__":
    main()
