import os, glob
import numpy as np
import pandas as pd
from pathlib import Path
from stats import friedman_test, nemenyi_posthoc, vargha_delaney_A12, cliffs_delta
from plotting import boxplot_metric, plot_convergence

def load_logs(root="data/logs"):
    rows = []
    for path in glob.glob(f"{root}/**/*.csv", recursive=True):
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        raise SystemExit("No log CSVs found. Run benchmarks first.")
    logs = pd.concat(rows, ignore_index=True)
    return logs

def main():
    logs = load_logs()
    # Final fitness per run (last iteration per seed)
    last = logs.sort_values(['algo','func','dim','seed','iter']).groupby(['algo','func','dim','seed']).tail(1)
    # Basic CSV summaries
    Path("analysis/out").mkdir(parents=True, exist_ok=True)
    last.to_csv("analysis/out/final_per_run.csv", index=False)

    # Friedman across algorithms per (func, dim)
    fr = friedman_test(last, value_col="best_fitness", group_cols=["func","dim"])
    fr.to_csv("analysis/out/friedman.csv", index=False)

    # Nemenyi p-value matrices saved as CSV per group
    for (keys, mat) in nemenyi_posthoc(last, value_col="best_fitness", group_cols=["func","dim"]):
        func, dim = keys
        mat.to_csv(f"analysis/out/nemenyi_{func}_D{dim}.csv")

    # Effect sizes for pairs (Vargha–Delaney and Cliff's delta)
    algos = sorted(last['algo'].unique())
    eff_rows = []
    for (func, dim), g in last.groupby(['func','dim']):
        for i in range(len(algos)):
            for j in range(i+1, len(algos)):
                a, b = algos[i], algos[j]
                xa = g[g['algo']==a]['best_fitness'].values
                xb = g[g['algo']==b]['best_fitness'].values
                # For minimization, transform so that "higher is better" is not required
                A12 = vargha_delaney_A12(-xa, -xb)
                cd = cliffs_delta(-xa, -xb)
                eff_rows.append({"func":func,"dim":dim,"A12(a>b)":A12,"cliffs_delta(a>b)":cd,"a":a,"b":b})
    eff = pd.DataFrame(eff_rows)
    eff.to_csv("analysis/out/effect_sizes.csv", index=False)

    # Boxplots
    boxplot_metric(last, "best_fitness", ["func","dim"], "analysis/figures")
    # Resource usage distributions (convert bytes to MB for plotting)
    last["mem_mb"] = last["mem_rss"]/ (1024*1024)
    boxplot_metric(last, "cpu_util_est", ["func","dim"], "analysis/figures")
    boxplot_metric(last, "mem_mb", ["func","dim"], "analysis/figures")

    # Convergence overlays (median per algo)
    plot_convergence(logs, "analysis/figures")

    print("Analysis complete. See analysis/out and analysis/figures")

if __name__ == "__main__":
    main()
