# NVIDIA GPT-OSS-120B Parity Runs

This setup runs ShinkaEvolve on the four FullReplacementMCTS comparison benchmarks:

- circle packing
- function minimization
- k-module problem
- signal processing

Set the NVIDIA NIM key in the shell before running. Do not commit the key.

```bash
export NVIDIA_API_KEY="<your-nvidia-nim-key>"
```

Install ShinkaEvolve in the environment used for experiments:

```bash
conda run -n openevolve_test pip install -e .
```

This workspace has been verified with an editable install in `openevolve_test`.

The Shinka configs use the OpenAI-compatible local backend string:

```text
local/openai/gpt-oss-120b@https://integrate.api.nvidia.com/v1?api_key_env=NVIDIA_API_KEY
```

## Smoke Tests

Run one generation per benchmark:

```bash
conda run -n openevolve_test python examples/openevolve_parity/run_parity.py \
  --config_path examples/openevolve_parity/circle_packing_nvidia.yaml \
  --generations 1 \
  --results_dir results/nvidia_smoke/circle_packing
```

Other benchmarks:

```bash
conda run -n openevolve_test python examples/openevolve_parity/run_parity.py --config_path examples/openevolve_parity/function_minimization_nvidia.yaml --generations 1 --results_dir results/nvidia_smoke/function_minimization
conda run -n openevolve_test python examples/openevolve_parity/run_parity.py --config_path examples/openevolve_parity/k_module_problem_nvidia.yaml --generations 1 --results_dir results/nvidia_smoke/k_module_problem
conda run -n openevolve_test python examples/openevolve_parity/run_parity.py --config_path examples/openevolve_parity/signal_processing_nvidia.yaml --generations 1 --results_dir results/nvidia_smoke/signal_processing
```

## Full Runs

Use the configs as committed:

- `circle_packing_nvidia.yaml`: 100 generations
- `function_minimization_nvidia.yaml`: 50 generations
- `k_module_problem_nvidia.yaml`: 50 generations
- `signal_processing_nvidia.yaml`: 50 generations

Outputs are written under the `results_dir` set in each YAML file:

- `results/openevolve_parity_circle_packing_nvidia`
- `results/openevolve_parity_function_minimization_nvidia`
- `results/openevolve_parity_k_module_problem_nvidia`
- `results/openevolve_parity_signal_processing_nvidia`

## Export Convergence

After a run, export a FullReplacementMCTS-style per-generation trace:

```bash
conda run -n openevolve_test python scripts/export_convergence.py \
  results/openevolve_parity_function_minimization_nvidia
```

This writes `results/.../convergence.jsonl` with `iteration` mapped from Shinka generation, `score`, `best_score`, `metrics`, correctness, timestamps, and program IDs. Synthetic island-copy rows are skipped by default; pass `--include-island-copies` if you want the raw database rows.

## Files Added

- `examples/openevolve_parity/run_parity.py`
- `examples/openevolve_parity/evaluate_adapter.py`
- `examples/openevolve_parity/*_nvidia.yaml`
- copied benchmark directories under `examples/openevolve_parity/`
- `scripts/export_convergence.py`
