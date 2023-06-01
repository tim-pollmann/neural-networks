import numpy as np
from .activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def __init__(self):
        pass

    def f(self, x):
        return np.tanh(x)

    def f_prime(self, x):
        return 1-np.tanh(x)**2
