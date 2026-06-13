#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("parity_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics"):
        metrics = dict(getattr(result, "metrics", {}) or {})
        artifacts = dict(getattr(result, "artifacts", {}) or {})
        return metrics, artifacts
    if isinstance(result, dict):
        result = dict(result)
        artifacts = result.pop("artifacts", {})
        if isinstance(result.get("metrics"), dict):
            metrics = dict(result["metrics"])
            artifacts = {**artifacts, **result.get("artifacts", {})}
            return metrics, artifacts
        return result, artifacts if isinstance(artifacts, dict) else {"artifacts": artifacts}
    if isinstance(result, (int, float)):
        return {"combined_score": float(result)}, {}
    return {"combined_score": 0.0, "error": f"Unexpected evaluator result type: {type(result).__name__}"}, {}


def _ensure_combined_score(metrics: dict[str, Any]) -> None:
    if isinstance(metrics.get("combined_score"), (int, float)) and not isinstance(metrics.get("combined_score"), bool):
        return
    for key in ("overall_score", "composite_score", "score", "fitness", "accuracy"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics["combined_score"] = float(value)
            metrics.setdefault("primary_metric_alias", key)
            return
    numeric = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    metrics["combined_score"] = float(sum(numeric) / len(numeric)) if numeric else 0.0
    if numeric:
        metrics.setdefault("primary_metric_alias", "numeric_average")


def _ensure_public_metrics(metrics: dict[str, Any]) -> None:
    public = metrics.get("public")
    if not isinstance(public, dict):
        public = {}

    for key, value in metrics.items():
        if key in {"public", "private", "text_feedback"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            public.setdefault(key, value)

    metrics["public"] = public


def main(program_path: str, results_dir: str, evaluator_path: str) -> None:
    evaluator = _load_module(Path(evaluator_path).resolve())
    if not hasattr(evaluator, "evaluate"):
        raise AttributeError(f"{evaluator_path} does not define evaluate(program_path)")

    try:
        metrics, artifacts = _normalize_result(evaluator.evaluate(program_path))
        _ensure_combined_score(metrics)
        _ensure_public_metrics(metrics)
        correct = float(metrics.get("combined_score", 0.0) or 0.0) > 0.0
        error = str(metrics.get("error", "")) or None
        if artifacts:
            metrics["private"] = {**metrics.get("private", {}), "artifacts": artifacts}
    except Exception as exc:  # noqa: BLE001
        metrics = {"combined_score": 0.0, "error": str(exc)}
        correct = False
        error = str(exc)

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--evaluator_path", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir, args.evaluator_path)
