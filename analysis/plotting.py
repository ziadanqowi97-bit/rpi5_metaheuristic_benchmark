import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def boxplot_metric(df, value_col, group_cols, outdir):
    os.makedirs(outdir, exist_ok=True)
    for (func, dim), g in df.groupby(group_cols):
        plt.figure()
        data = [g[g['algo']==a][value_col].values for a in sorted(g['algo'].unique())]
        labels = sorted(g['algo'].unique())
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.yscale('log')  # fitness can span orders of magnitude
        plt.ylabel(value_col)
        plt.title(f"{value_col} by algo — {func}, D={dim}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"box_{value_col}_{func}_D{dim}.png"))
        plt.close()

def plot_convergence(df, outdir, max_iters=None):
    os.makedirs(outdir, exist_ok=True)
    # Expect per-iteration logs; we'll plot median convergence per algo
    for (func, dim), g in df.groupby(['func','dim']):
        plt.figure()
        for algo, ga in g.groupby('algo'):
            med = ga.groupby('iter')['best_fitness'].median()
            if max_iters is not None:
                med = med.iloc[:max_iters]
            plt.plot(med.index.values, med.values, label=algo)
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Median best fitness')
        plt.title(f"Median convergence — {func}, D={dim}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"convergence_{func}_D{dim}.png"))
        plt.close()
