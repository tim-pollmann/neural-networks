from .layer import Layer
import numpy as np


class FCLayer(Layer):
    def __init__(self, input_size, output_size, activation_function):
        super().__init__(input_size, output_size, activation_function)

        self.a = None
        self.z = None
        self.x = None
        assert input_size[0] == 1 and input_size[2] == 1
        assert output_size[0] == 1 and output_size[2] == 1
        
        self.W = np.random.rand(output_size[1], input_size[1]) - 0.5
        self.b = np.random.rand(output_size[1], 1) - 0.5

    def forward_propagation(self, x):
        self.x = x[0]
        self.z = np.dot(self.W, self.x) + self.b
        self.a = self.activation_function.f(self.z)
        return self.a[np.newaxis]

    def backward_propagation(self, e, alpha):
        e = np.dot(self.activation_function.f_prime(self.z), e)
        de_dx = np.dot(self.W.T, e)
        de_dW = np.dot(e, self.x.T)
        de_db = self.b * e

        self.W -= alpha * de_dW
        self.b -= alpha * de_db

        return de_dx
