import os, csv, time, yaml, argparse, math, numpy as np, psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from .functions import FUNCS
from .algorithms.gwo import GWO
from .algorithms.pso import PSO
from .algorithms.ga import GA
from .algorithms.de import DE
from .algorithms.hs import HS
from .utils import set_single_core_affinity, IterationMonitor, read_cpu_temp, read_cpu_freq

ALGOS = {"gwo": GWO, "pso": PSO, "ga": GA, "de": DE, "hs": HS}

def build_algo(name, func, dim, bounds, pop_size, rng, max_evals):
    if name == "gwo":
        return GWO(func, dim, bounds, pop_size, rng, max_evals)
    if name == "pso":
        return PSO(func, dim, bounds, pop_size, rng, max_evals)
    if name == "ga":
        return GA(func, dim, bounds, pop_size, rng, max_evals)
    if name == "de":
        return DE(func, dim, bounds, pop_size, rng, max_evals)
    if name == "hs":
        return HS(func, dim, bounds, pop_size, rng, max_evals)
    raise ValueError(f"Unknown algo {name}")

def ensure_dirs():
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/summary").mkdir(parents=True, exist_ok=True)

def run_one(algo_name: str, func_name: str, dim: int, cfg: Dict[str, Any], seed: int):
    func = FUNCS[func_name]
    bounds = cfg["bounds"][func_name]
    pop = int(cfg["pop_size"])
    budget = int(cfg["eval_budget"])
    iterations = max(1, budget // pop)
    log_every = int(cfg.get("log_freq_iters", 1))

    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # for libs expecting global RNG

    algo = build_algo(algo_name, func, dim, bounds, pop, rng, budget)
    mon = IterationMonitor(log_every)

    outdir = Path(f"data/logs/{algo_name}/{func_name}/D{dim}")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"run_{seed}.csv"
    if outfile.exists():
        return  # skip if already done

    fields = ["timestamp","algo","func","dim","seed","iter","evals","best_fitness",
              "cpu_util_est","mem_rss","cpu_freq_mhz","cpu_temp_c"]
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)

        def log_callback(it, evals, best):
            if (it % log_every) != 0:
                return
            snap = mon.snapshot()
            cpu_freq = read_cpu_freq() if cfg.get("enable_freq_temp", True) else float("nan")
            cpu_temp = read_cpu_temp() if cfg.get("enable_freq_temp", True) else float("nan")
            w.writerow([datetime.utcnow().isoformat(), algo_name, func_name, dim, seed, it, evals, best,
                        snap["cpu_util_est"], snap["mem_rss"], cpu_freq, cpu_temp])

        algo.run(iterations, log_callback=log_callback)

    # index for resume
    with open("data/summary/runs_index.csv", "a", newline="") as fsum:
        wsum = csv.writer(fsum)
        wsum.writerow([algo_name, func_name, dim, seed, str(outfile)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke test")
    parser.add_argument("--core", type=int, default=None, help="Set process CPU affinity to this core id")
    parser.add_argument("--config", type=str, default="benchmarks/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ensure_dirs()

    if args.core is not None:
        set_single_core_affinity(args.core)

    algos = cfg["algorithms"]
    funcs = cfg["functions"]
    dims = cfg["dims"]
    reps = int(cfg["repetitions"])
    budget = int(cfg["eval_budget"])
    seed_base = int(cfg.get("seed_base", 42))

    if args.smoke:
        reps = 1
        cfg["eval_budget"] = 300  # tiny

    # Create/clear runs index if not present
    idxfile = Path("data/summary/runs_index.csv")
    if not idxfile.exists():
        with open(idxfile, "w", newline="") as fsum:
            csv.writer(fsum).writerow(["algo","func","dim","seed","path"])

    total = 0
    for algo in algos:
        for func in funcs:
            for dim in dims:
                for r in range(reps):
                    seed = seed_base + (total % 1000000)  # deterministic sequence
                    run_one(algo, func, dim, cfg, seed)
                    total += 1

if __name__ == "__main__":
    main()
