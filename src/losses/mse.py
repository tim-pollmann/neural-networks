import numpy as np
from .loss import Loss


class MSE(Loss):
    def __init__(self, shape):
        super().__init__(shape)

    def f(self, d, y):
        assert d.shape == self.shape and y.shape == self.shape
        return np.mean(np.power(d[0]-y[1], 2))

    def f_prime(self, d, y):
        assert d.shape == self.shape and y.shape == self.shape
        return 2*(y[0]-d[0])/d[0].size
