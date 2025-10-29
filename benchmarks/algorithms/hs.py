import numpy as np
from .base import OptimizerBase

class HS(OptimizerBase):
    name = "hs"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals, HMCR=0.9, PAR=0.1, bw=0.01):
        super().__init__(func, dim, bounds, pop_size, rng, max_evals)
        self.HMCR = HMCR  # harmony memory considering rate
        self.PAR = PAR    # pitch adjusting rate
        self.bw = bw      # bandwidth for pitch adjustment

    def step(self):
        NP, D = self.X.shape
        X_new = np.empty_like(self.X)
        for i in range(NP):
            x = np.empty(D)
            for d in range(D):
                if self.rng.random() < self.HMCR:
                    # choose from harmony memory
                    x[d] = self.X[self.rng.integers(0, NP), d]
                    if self.rng.random() < self.PAR:
                        # pitch adjustment
                        delta = self.bw * (self.upper[d] - self.lower[d])
                        x[d] += self.rng.uniform(-delta, delta)
                else:
                    x[d] = self.rng.uniform(self.lower[d], self.upper[d])
            X_new[i] = np.clip(x, self.lower, self.upper)
        F_new = self._evaluate(X_new)
        # replace worst individuals with improved ones
        for i in range(NP):
            j = np.argmax(self.F)
            if F_new[i] < self.F[j]:
                self.X[j] = X_new[i]
                self.F[j] = F_new[i]
