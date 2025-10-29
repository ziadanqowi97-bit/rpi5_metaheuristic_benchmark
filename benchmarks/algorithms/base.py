import numpy as np

class OptimizerBase:
    name = "base"

    def __init__(self, func, dim, bounds, pop_size, rng, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.pop_size = pop_size
        self.rng = rng
        self.max_evals = int(max_evals)
        self.eval_count = 0

        self.lower = self.bounds[0] * np.ones(self.dim)
        self.upper = self.bounds[1] * np.ones(self.dim)

        # Population and fitness
        self.X = self.rng.uniform(self.lower, self.upper, size=(self.pop_size, self.dim))
        self.F = self._evaluate(self.X)

    def _evaluate(self, X):
        vals = self.func(X)
        self.eval_count += X.shape[0]
        return vals

    def best(self):
        idx = np.argmin(self.F)
        return self.F[idx], self.X[idx].copy()

    def step(self):
        raise NotImplementedError

    def run(self, iterations, log_callback=None):
        for it in range(iterations):
            self.step()
            if log_callback is not None:
                fbest, _ = self.best()
                log_callback(it=it+1, evals=self.eval_count, best=fbest)
            if self.eval_count >= self.max_evals:
                break
