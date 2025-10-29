import numpy as np
from .base import OptimizerBase

class GA(OptimizerBase):
    name = "ga"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals, pc=0.9, pm=0.1, eta=10):
        super().__init__(func, dim, bounds, pop_size, rng, max_evals)
        self.pc = pc
        self.pm = pm
        self.eta = eta  # SBX parameter

    def _tournament(self, k=2):
        idx = self.rng.integers(0, self.pop_size, size=(self.pop_size, k))
        best = idx[np.arange(self.pop_size), np.argmin(self.F[idx], axis=1)]
        return self.X[best]

    def _sbx_crossover(self, parents):
        # parents shape (pop, D)
        u = self.rng.random(parents.shape)
        beta = np.where(u <= 0.5, (2*u)**(1/(self.eta+1)), (1/(2*(1-u)))**(1/(self.eta+1)))
        child1 = 0.5*((1+beta)*parents[0::2] + (1-beta)*parents[1::2])
        child2 = 0.5*((1-beta)*parents[0::2] + (1+beta)*parents[1::2])
        return np.concatenate([child1, child2], axis=0)

    def _gaussian_mutation(self, X, sigma=0.1):
        noise = self.rng.normal(0, sigma, size=X.shape)*(self.upper - self.lower)
        mask = self.rng.random(X.shape) < self.pm
        Xm = np.where(mask, X + noise, X)
        return np.clip(Xm, self.lower, self.upper)

    def step(self):
        # Selection
        parents = self._tournament(k=2)
        # Crossover (pairwise)
        if self.pc > 0:
            # Ensure even number for pairing
            if parents.shape[0] % 2 == 1:
                parents = parents[:-1]
            offspring = self._sbx_crossover(parents)
        else:
            offspring = parents.copy()

        # Mutation
        offspring = self._gaussian_mutation(offspring, sigma=0.1)

        # Evaluate offspring (equal cost: evaluate pop_size solutions)
        offspring = offspring[:self.pop_size]
        F_off = self._evaluate(offspring)

        # Survivor selection (elitist)
        X_all = np.vstack([self.X, offspring])
        F_all = np.concatenate([self.F, F_off])
        idx = np.argsort(F_all)[:self.pop_size]
        self.X = X_all[idx]
        self.F = F_all[idx]
