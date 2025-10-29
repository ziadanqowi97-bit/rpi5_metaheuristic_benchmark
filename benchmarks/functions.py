import numpy as np

def sphere(x):
    return np.sum(x*x, axis=-1)

def rosenbrock(x):
    # x shape (..., D)
    x1 = x[..., :-1]
    x2 = x[..., 1:]
    return np.sum(100.0 * (x2 - x1**2.0)**2.0 + (1 - x1)**2.0, axis=-1)

def rastrigin(x):
    return np.sum(x*x - 10.0*np.cos(2*np.pi*x) + 10.0, axis=-1)

def ackley(x):
    # standard Ackley (a=20, b=0.2, c=2π)
    a = 20.0; b = 0.2; c = 2*np.pi
    d = x.shape[-1]
    sum_sq = np.sum(x*x, axis=-1) / d
    sum_cos = np.sum(np.cos(c*x), axis=-1) / d
    return -a*np.exp(-b*np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.e

FUNCS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}
