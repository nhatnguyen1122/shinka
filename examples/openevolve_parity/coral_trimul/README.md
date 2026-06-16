# CORAL TriMul Kernel Engineering

This benchmark wraps CORAL `examples/kernel_engineering/trimul` as a CUDA/Triton optimization task. It requires a Linux server with an NVIDIA CUDA GPU, PyTorch CUDA, Triton, and PyYAML.

Install into your experiment environment:

```bash
conda create -n coral_kernel python=3.11 -y
conda activate coral_kernel
pip install -r examples/openevolve_parity/coral_trimul/requirements.txt
```

Run ShinkaEvolve on the CORAL TriMul task:

```bash
export NVIDIA_API_KEY=...
export CORAL_PYTHON="$(which python)"
python scripts/run_nvidia_parity_matrix.py \
  --runs 5 \
  --benchmarks trimul \
  --generations 300 \
  --output-root results/coral_trimul_shinka_5runs_300gen
```

For a cheap server smoke test, prefix the command with `CORAL_TRIMUL_QUICK=1` and use `--runs 1 --generations 1 --fail-fast`. Do not use quick mode for final comparisons.
