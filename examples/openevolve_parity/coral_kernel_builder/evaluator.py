from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKDATA = ROOT / "taskdata"

BASELINE_CYCLES = 147_734
BEST_KNOWN_CYCLES = 1_363
REAL_PARAMS = {"forest_height": 10, "rounds": 16, "batch_size": 256, "iterations": 8}
TUNE_PARAMS = {"forest_height": 10, "rounds": 4, "batch_size": 64, "iterations": 2}


def _python_executable() -> str:
    return os.environ.get("CORAL_PYTHON") or sys.executable


def _run_evaluation(program_path: Path, timeout: int, params: dict[str, int]) -> dict:
    frozen_path = TASKDATA / "frozen_problem.py"
    script = textwrap.dedent(
        f"""\
        import json
        import os
        import sys

        sys.path.insert(0, os.path.dirname({str(frozen_path.resolve())!r}))
        from frozen_problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES

        source = open({str(program_path.resolve())!r}, encoding="utf-8").read()
        namespace = {{"__name__": "__main__"}}
        exec(source, namespace)

        if "KernelBuilder" not in namespace:
            print(json.dumps({{"error": "KernelBuilder class not found"}}))
            sys.exit(0)

        KernelBuilder = namespace["KernelBuilder"]

        def do_kernel_test(forest_height, rounds, batch_size):
            forest = Tree.generate(forest_height)
            inp = Input.generate(forest, batch_size, rounds)
            mem = build_mem_image(forest, inp)

            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

            machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
            machine.enable_pause = False
            machine.enable_debug = False
            machine.run()

            for ref_mem in reference_kernel2(mem):
                pass

            inp_values_p = ref_mem[6]
            actual = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            expected = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
            if actual != expected:
                return machine.cycle, False, "Incorrect output values"
            return machine.cycle, True, ""

        for i in range({params["iterations"]}):
            cycles, is_correct, error_msg = do_kernel_test(
                {params["forest_height"]}, {params["rounds"]}, {params["batch_size"]}
            )
            if not is_correct:
                print(json.dumps({{
                    "cycles": cycles,
                    "is_correct": False,
                    "error_msg": f"Iteration {{i}}: {{error_msg}}",
                }}))
                sys.exit(0)

        print(json.dumps({{"cycles": cycles, "is_correct": True, "error_msg": ""}}))
        """
    )
    proc = subprocess.run(
        [_python_executable(), "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-2000:] or f"returncode={proc.returncode}"}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": f"No JSON result. stdout={proc.stdout[-500:]} stderr={proc.stderr[-500:]}"}


def evaluate(program_path: str) -> dict:
    path = Path(program_path).resolve()
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return {"combined_score": 0.0, "correct": 0.0, "error": f"Python syntax check failed: {exc.msg}"}

    if os.environ.get("CORAL_KERNEL_BUILDER_COMPILE_ONLY") == "1":
        return {"combined_score": 0.0, "compile_ok": 1.0, "note": "CORAL_KERNEL_BUILDER_COMPILE_ONLY=1"}

    params = TUNE_PARAMS if os.environ.get("CORAL_KERNEL_BUILDER_TUNE") == "1" else REAL_PARAMS
    timeout = int(os.environ.get("CORAL_KERNEL_BUILDER_TIMEOUT", "120"))
    try:
        result = _run_evaluation(path, timeout, params)
    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "correct": 0.0, "error": f"Evaluation timed out after {timeout}s"}

    if "error" in result:
        return {"combined_score": 0.0, "correct": 0.0, "error": result["error"]}

    cycles = int(result.get("cycles", BASELINE_CYCLES * 2))
    correct = bool(result.get("is_correct", False))
    if not correct:
        return {
            "combined_score": 0.0,
            "correct": 0.0,
            "cycles": cycles,
            "error": result.get("error_msg", "incorrect output"),
        }

    combined_score = BASELINE_CYCLES / cycles if cycles > 0 else 0.0
    return {
        "combined_score": float(combined_score),
        "correct": 1.0,
        "cycles": cycles,
        "cycle_count": cycles,
        "speedup_vs_baseline": float(combined_score),
        "baseline_cycles": BASELINE_CYCLES,
        "best_known_cycles": BEST_KNOWN_CYCLES,
        "tune_mode": 1.0 if os.environ.get("CORAL_KERNEL_BUILDER_TUNE") == "1" else 0.0,
        "lower_cycles_is_better": 1.0,
    }
