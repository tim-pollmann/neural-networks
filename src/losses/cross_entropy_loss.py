import numpy as np
from .loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self, shape):
        super().__init__(shape)

    def f(self, d, y):
        assert d.shape == self.shape and y.shape == self.shape, f"{d.shape}, {y.shape}, {self.shape}"
        return - np.sum(d[0]*np.log(y[0]))

    def f_prime(self, d, y):
        assert d.shape == self.shape and y.shape == self.shape
        return -d[0]/y[0]
