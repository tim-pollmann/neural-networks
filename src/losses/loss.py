class Loss:
    def __init__(self, shape):
        self.shape = shape

    def f(self, d, y):
        raise NotImplementedError

    def f_prime(self, d, y):
        raise NotImplementedError
