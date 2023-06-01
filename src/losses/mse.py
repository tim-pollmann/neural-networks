import numpy as np
from .loss import Loss


class MSE(Loss):
    def __init__(self):
        pass

    def f(self, d, y):
        return np.mean(np.power(d-y, 2))

    def f_prime(self, d, y):
        return 2*(y-d)/d.size
