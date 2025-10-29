# RPi 5 Metaheuristics Benchmark (GWO, PSO, GA, DE, HS)

End-to-end scaffold to compare five metaheuristics on Raspberry Pi 5 across Sphere, Rosenbrock, Rastrigin, Ackley with D ∈ {10, 30, 50}. 
It enforces *budget parity*, *Big‑O parity* (same population size and function evaluations per iteration), *single-core execution*, 
and logs per-iteration CPU/RAM plus optional CPU frequency and temperature (no external sensors).

> ⚠️ Run-time can be long if you keep 200× repetitions per (function × dim × algorithm). Start with fewer repetitions to smoke-test, then scale.

## 0) Setup (on Raspberry Pi 5)
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv taskset util-linux
python3 -m venv ~/.venvs/rpi5bench
source ~/.venvs/rpi5bench/bin/activate
pip install -U pip
pip install -r requirements.txt
```
Force single-threaded math libs and pin to 1 core:
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Pin to CPU core 3 (change as you wish)
taskset -c 3 python benchmarks/runner.py --help
```
Optionally fix CPU governor and read-only freq:
```bash
# View available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
# Set to 'performance' for stability (requires sudo)
sudo bash scripts/governor_setup.sh performance
```

## 1) Quick smoke test
```bash
# One repetition per combo, tiny budget
taskset -c 3 python benchmarks/runner.py --smoke
```

## 2) Full experiment
Edit `benchmarks/config.yaml` to set repetitions, pop_size, eval_budget, seeds, etc. Then:
```bash
taskset -c 3 python benchmarks/runner.py
```

Logs appear under `data/logs/<algo>/<func>/D<dim>/run_<seed>.csv`.
A master index `data/summary/runs_index.csv` is kept for resuming.

## 3) Analysis (statistics + plots)
```bash
python analysis/analyze.py
```
Outputs:
- Friedman/Nemenyi tables (CSV)
- Effect sizes (Vargha–Delaney A12, Cliff's δ)
- Boxplots and convergence overlays in `analysis/figures/`

## Design choices
- **Budget parity**: Each run uses the same total function-evaluation budget. Iterations = `eval_budget // pop_size` across all algorithms.
- **Big‑O parity**: All algorithms maintain `O(pop_size × dimension)` per iteration; mutation/crossover are vectorized.
- **Initialization parity**: Same seeding strategy, same uniform sampling bounds per problem.
- **Tuning cost parity**: Optional small offline tuning with equal evaluation budget per algorithm. See `benchmarks/tune.py`.

## Recover/Resume
`runner.py` skips runs that already have complete CSV logs and are indexed in `data/summary/runs_index.csv`.
Terminate safely with Ctrl+C; re-run to resume.

## CPU freq/temperature without sensors
We try (in order): `vcgencmd` if available, then `/sys/class/thermal/*/temp` and `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`.
If unavailable, the fields will be NaN gracefully.

---

**License**: MIT
