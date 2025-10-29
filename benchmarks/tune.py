# Optional, equal-cost offline tuning with a small budget per algorithm.
import os, csv, yaml, argparse, numpy as np
from pathlib import Path
from .functions import FUNCS
from .algorithms.gwo import GWO
from .algorithms.pso import PSO
from .algorithms.ga import GA
from .algorithms.de import DE
from .algorithms.hs import HS

ALGOS = {"gwo": GWO, "pso": PSO, "ga": GA, "de": DE, "hs": HS}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if not cfg["tuning"]["enabled"]:
        print("Tuning disabled in config.")
        return

    out = {}
    # Define tiny hyper grids (same #configs per algo)
    grids = {
        "gwo": [{"a_decay": a} for a in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]],
        "pso": [{"w": w, "c1": 1.2, "c2": 1.8} for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
        "ga":  [{"pc": pc, "pm": 0.1, "eta": 10} for pc in [0.6, 0.7, 0.8, 0.9, 1.0, 0.5]],
        "de":  [{"F": F, "CR": 0.9} for F in [0.3, 0.5, 0.7, 0.9, 0.2, 0.4]],
        "hs":  [{"HMCR": h, "PAR": 0.1, "bw": 0.01} for h in [0.7, 0.8, 0.9, 0.95, 0.6, 0.85]],
    }
    dims = [10]
    func = FUNCS["sphere"]
    bounds = cfg["bounds"]["sphere"]
    pop = cfg["pop_size"]
    reps_per = cfg["tuning"]["reps_per_config"]
    budget = cfg["tuning"]["eval_budget_per_rep"]

    for algo_name, grid in grids.items():
        best_med = float("inf")
        best_params = {}
        for i, params in enumerate(grid):
            vals = []
            for r in range(reps_per):
                seed = cfg["seed_base"] + r + i*100
                rng = np.random.default_rng(seed)
                if algo_name == "gwo":
                    algo = GWO(func, dims[0], bounds, pop, rng, budget, **params)
                elif algo_name == "pso":
                    algo = PSO(func, dims[0], bounds, pop, rng, budget, **params)
                elif algo_name == "ga":
                    algo = GA(func, dims[0], bounds, pop, rng, budget, **params)
                elif algo_name == "de":
                    algo = DE(func, dims[0], bounds, pop, rng, budget, **params)
                elif algo_name == "hs":
                    algo = HS(func, dims[0], bounds, pop, rng, budget, **params)
                iterations = max(1, budget // pop)
                algo.run(iterations)
                fbest, _ = algo.best()
                vals.append(fbest)
            med = float(np.median(vals))
            if med < best_med:
                best_med = med
                best_params = params
        out[algo_name] = best_params

    Path("data/params").mkdir(parents=True, exist_ok=True)
    with open("data/params/tuned_params.json", "w") as f:
        import json
        json.dump(out, f, indent=2)
    print("Saved tuned params -> data/params/tuned_params.json")

if __name__ == "__main__":
    main()
