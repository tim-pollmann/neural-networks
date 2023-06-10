import numpy as np
from .activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__()

    def f(self, x):
        return np.tanh(x)

    def f_prime(self, x):
        n = len(x)
        p = 1-np.tanh(x)**2
        result = np.zeros([n, n])
        for i in range(n):
                result[i, i] = p[i]
        return result

