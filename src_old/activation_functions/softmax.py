import numpy as np
from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def f(self, x):
        exps = np.exp(x)
        return exps/np.sum(exps)

    def f_prime(self, x):
        n = len(x)
        p = self.f(x)
        result = np.empty([n, n])
        for i in range(n):
            for j in range(n):
                result[i, j] = p[i] * ((i == j) - p[j])
        return result
