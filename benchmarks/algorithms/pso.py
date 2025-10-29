import numpy as np
from .base import OptimizerBase

class PSO(OptimizerBase):
    name = "pso"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals, w=0.729, c1=1.49445, c2=1.49445, vmax_frac=0.2):
        super().__init__(func, dim, bounds, pop_size, rng, max_evals)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.V = np.zeros_like(self.X)
        self.P = self.X.copy()
        self.PF = self.F.copy()
        self.gbest = self.X[np.argmin(self.F)].copy()
        self.vmax = (self.upper - self.lower) * vmax_frac

    def step(self):
        r1 = self.rng.random(self.X.shape)
        r2 = self.rng.random(self.X.shape)
        self.V = self.w*self.V + self.c1*r1*(self.P - self.X) + self.c2*r2*(self.gbest - self.X)
        self.V = np.clip(self.V, -self.vmax, self.vmax)
        X_new = self.X + self.V
        X_new = np.clip(X_new, self.lower, self.upper)

        F_new = self._evaluate(X_new)

        improved = F_new < self.F
        self.X[improved] = X_new[improved]
        self.F[improved] = F_new[improved]

        p_improved = F_new < self.PF
        self.P[p_improved] = X_new[p_improved]
        self.PF[p_improved] = F_new[p_improved]

        self.gbest = self.P[np.argmin(self.PF)].copy()
