import numpy as np
from .base import OptimizerBase

class GWO(OptimizerBase):
    name = "gwo"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals, a_decay=2.0):
        super().__init__(func, dim, bounds, pop_size, rng, max_evals)
        self.a0 = 2.0
        self.a_decay = a_decay
        self.iteration = 0
        self.max_iterations = max(1, self.max_evals // self.pop_size)

    def step(self):
        self.iteration += 1
        a = self.a0 - (self.a0 * self.iteration / self.max_iterations)

        # Identify alpha, beta, delta
        idx = np.argsort(self.F)
        X_alpha, X_beta, X_delta = self.X[idx[0]], self.X[idx[1]], self.X[idx[2]]

        A1 = 2*a*self.rng.random((self.pop_size, self.dim)) - a
        C1 = 2*self.rng.random((self.pop_size, self.dim))
        D_alpha = np.abs(C1*X_alpha - self.X)
        X1 = X_alpha - A1*D_alpha

        A2 = 2*a*self.rng.random((self.pop_size, self.dim)) - a
        C2 = 2*self.rng.random((self.pop_size, self.dim))
        D_beta = np.abs(C2*X_beta - self.X)
        X2 = X_beta - A2*D_beta

        A3 = 2*a*self.rng.random((self.pop_size, self.dim)) - a
        C3 = 2*self.rng.random((self.pop_size, self.dim))
        D_delta = np.abs(C3*X_delta - self.X)
        X3 = X_delta - A3*D_delta

        X_new = (X1 + X2 + X3) / 3.0
        X_new = np.clip(X_new, self.lower, self.upper)

        F_new = self._evaluate(X_new)
        improved = F_new < self.F
        self.X[improved] = X_new[improved]
        self.F[improved] = F_new[improved]
