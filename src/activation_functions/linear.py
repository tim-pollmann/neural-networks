import numpy as np
from .activation_function import ActivationFunction


class Linear(ActivationFunction):
    def __init__(self):
        pass

    def f(self, x):
        return x

    def f_prime(self, x):
        return np.identity(len(x))
