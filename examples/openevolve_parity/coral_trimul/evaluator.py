from __future__ import annotations

import math
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TASKDATA = ROOT / "taskdata"


def _python_executable() -> str:
    configured = os.environ.get("CORAL_PYTHON")
    if configured:
        return configured
    return "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable


def _parse_popcorn_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()
    return parsed


def _write_specs(task_yml: Path, mode: str, output_path: Path) -> None:
    config = yaml.safe_load(task_yml.read_text(encoding="utf-8")) or {}
    key = "tests" if mode == "test" else "benchmarks"
    specs = list(config.get(key, []))
    if os.environ.get("CORAL_TRIMUL_QUICK") == "1":
        specs = specs[:1]
    with output_path.open("w", encoding="utf-8") as f:
        for spec in specs:
            f.write("; ".join(f"{k}: {v}" for k, v in spec.items()) + "\n")


def _run_harness(program_path: Path, mode: str, timeout: int) -> tuple[dict[str, str], str, str, int]:
    with tempfile.TemporaryDirectory(prefix="coral_trimul_") as tmp:
        tmpdir = Path(tmp)
        for source in TASKDATA.iterdir():
            if source.is_file():
                shutil.copy2(source, tmpdir / source.name)
        shutil.copy2(program_path, tmpdir / "submission.py")
        spec_path = tmpdir / f"{mode}.txt"
        _write_specs(tmpdir / "task.yml", mode, spec_path)

        read_fd, write_fd = os.pipe()
        env = os.environ.copy()
        env["POPCORN_FD"] = str(write_fd)
        proc = subprocess.Popen(
            [_python_executable(), "eval.py", mode, spec_path.name],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        with os.fdopen(read_fd, "r", encoding="utf-8") as reader:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                popcorn = reader.read()
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                popcorn = reader.read()
                return {"check": "fail", "error": f"{mode} timed out after {timeout}s"}, stdout, stderr, proc.returncode or 124
        return _parse_popcorn_output(popcorn), stdout, stderr, proc.returncode


def _extract_errors(results: dict[str, str]) -> str:
    errors = [value for key, value in results.items() if key.endswith(".error")]
    return "; ".join(errors[:3]) if errors else results.get("error", "unknown error")


def _geomean(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def evaluate(program_path: str) -> dict:
    path = Path(program_path).resolve()
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return {"combined_score": 0.0, "error": f"Python syntax check failed: {exc.msg}"}

    if os.environ.get("CORAL_TRIMUL_COMPILE_ONLY") == "1":
        return {"combined_score": 0.0, "compile_ok": 1.0, "note": "CORAL_TRIMUL_COMPILE_ONLY=1"}

    if not TASKDATA.exists():
        return {"combined_score": 0.0, "error": f"Missing CORAL taskdata directory: {TASKDATA}"}

    timeout = int(os.environ.get("CORAL_TRIMUL_TIMEOUT", "1200"))
    test_results, test_stdout, test_stderr, test_rc = _run_harness(path, "test", timeout)
    if test_results.get("check") != "pass":
        return {
            "combined_score": 0.0,
            "correct": 0.0,
            "error": f"Correctness failed: {_extract_errors(test_results)}",
            "returncode": test_rc,
            "stderr": test_stderr[-2000:],
            "stdout": test_stdout[-2000:],
        }

    if os.environ.get("CORAL_TRIMUL_TEST_ONLY") == "1":
        return {"combined_score": 1.0, "correct": 1.0, "note": "CORAL_TRIMUL_TEST_ONLY=1"}

    bench_results, bench_stdout, bench_stderr, bench_rc = _run_harness(path, "leaderboard", timeout)
    if bench_results.get("check") != "pass":
        return {
            "combined_score": 0.0,
            "correct": 1.0,
            "error": f"Benchmark failed: {_extract_errors(bench_results)}",
            "returncode": bench_rc,
            "stderr": bench_stderr[-2000:],
            "stdout": bench_stdout[-2000:],
        }

    timings_ns = []
    for index in range(int(bench_results.get("benchmark-count", "0"))):
        value = bench_results.get(f"benchmark.{index}.mean")
        if value is not None:
            timings_ns.append(float(value))
    geomean_ns = _geomean(timings_ns)
    geomean_us = geomean_ns / 1000.0 if geomean_ns > 0 else 0.0
    score = 1000.0 / geomean_us if geomean_us > 0 else 0.0
    return {
        "combined_score": float(score),
        "correct": 1.0,
        "runtime_us": float(geomean_us),
        "runtime_ns_geomean": float(geomean_ns),
        "benchmark_count": len(timings_ns),
        "quick_mode": 1.0 if os.environ.get("CORAL_TRIMUL_QUICK") == "1" else 0.0,
    }
