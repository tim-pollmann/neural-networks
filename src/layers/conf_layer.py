from .layer import Layer
import numpy as np


class ConfLayer(Layer):
    def __init__(self, input_size, output_size, n_kernels, kernel_size, activation_function):
        super().__init__(input_size, output_size, activation_function)
        self.kernels = np.random.rand([n_kernels, kernel_size, kernel_size]) - 0.5

    def forward_propagation(self, x):
        self.x = x
        self.z = [self.conf2d(x, kernel) for kernel in self.kernels]
        self.a = self.activation_function.f(self.z)
        return self.a

    def backward_propagation(self, e, alpha):
        e = np.dot(self.activation_function.f_prime(self.z), e)
        de_dx = np.dot(self.W.T, e)
        de_dW = np.dot(e, self.x.T)
        de_db = self.b * e

        self.W -= alpha * de_dW
        self.b -= alpha * de_db

        return de_dx
    
    def conf2d(matrix, kernel):
        pass
