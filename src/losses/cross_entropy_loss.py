import numpy as np
from .loss import Loss


class CrossEntropyLoss(Loss):
    def __init__(self):
        pass

    def f(self, d, y):
        return - np.sum(d*np.log(y[0]))

    def f_prime(self, d, y):
        return -d/y[0]
