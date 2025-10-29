import numpy as np
from .base import OptimizerBase

class DE(OptimizerBase):
    name = "de"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals, F=0.5, CR=0.9, strategy="rand/1/bin"):
        super().__init__(func, dim, bounds, pop_size, rng, max_evals)
        self.FF = F
        self.CR = CR
        self.strategy = strategy

    def step(self):
        X = self.X
        NP, D = X.shape
        idxs = np.arange(NP)
        X_new = np.empty_like(X)

        for i in range(NP):
            a, b, c = self.rng.choice(idxs[idxs != i], 3, replace=False)
            if self.strategy.startswith("best/1"):
                best = X[np.argmin(self.F)]
                v = best + self.FF*(X[a]-X[b])
            else:  # rand/1
                v = X[a] + self.FF*(X[b]-X[c])

            jrand = self.rng.integers(0, D)
            cross = self.rng.random(D) < self.CR
            cross[jrand] = True
            u = np.where(cross, v, X[i])
            X_new[i] = np.clip(u, self.lower, self.upper)

        F_new = self._evaluate(X_new)
        improved = F_new < self.F
        self.X[improved] = X_new[improved]
        self.F[improved] = F_new[improved]
