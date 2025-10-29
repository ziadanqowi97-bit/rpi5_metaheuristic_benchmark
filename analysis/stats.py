import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from itertools import combinations

def friedman_test(df, value_col, group_cols, algo_col="algo"):
    # df contains rows per run with columns group_cols (e.g., func, dim), algo and a numeric value
    results = []
    for keys, g in df.groupby(group_cols):
        # pivot to shape (n_subjects, n_algos) where subjects = repetitions/ seeds
        pivot = g.pivot_table(index="seed", columns=algo_col, values=value_col, aggfunc="median")
        pivot = pivot.dropna(axis=0, how="any")  # remove seeds with missing algos
        if pivot.shape[0] < 2:
            continue
        stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        results.append({"group": keys, "k_algos": pivot.shape[1], "n_blocks": pivot.shape[0], "stat": stat, "pvalue": p})
    return pd.DataFrame(results)

def nemenyi_posthoc(df, value_col, group_cols, algo_col="algo"):
    rows = []
    for keys, g in df.groupby(group_cols):
        pivot = g.pivot_table(index="seed", columns=algo_col, values=value_col, aggfunc="median")
        pivot = pivot.dropna(axis=0, how="any")
        if pivot.shape[0] < 2:
            continue
        pvals = sp.posthoc_nemenyi(pivot)
        rows.append((keys, pvals))
    return rows

def vargha_delaney_A12(x, y):
    # Probability that a random x > a random y (for minimization use reversed sense accordingly)
    nx = len(x); ny = len(y)
    ranks = pd.Series(np.concatenate([x, y])).rank()
    rx = ranks[:nx].sum()
    A = (rx/nx - (nx+1)/2) / ny
    return float(A)

def cliffs_delta(x, y):
    import math
    n = len(x); m = len(y)
    greater = 0; less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    d = (greater - less)/(n*m)
    return float(d)
