from .layer import Layer
import numpy as np


class ConvLayer(Layer):
    def __init__(self, input_size, output_size, n_kernels, kernel_size, activation_function):
        super().__init__(input_size, output_size, activation_function)
        self.a = None
        self.z = None
        self.x = None
        self.kernel_size = kernel_size
        # self.kernels = np.random.rand([n_kernels, kernel_size, kernel_size]) - 0.5

    def forward_propagation(self, x):
        self.x = x
        self.z = [self.conf2d(x, kernel) for kernel in self.kernels]
        self.a = self.activation_function.f(self.z)
        return self.a

    def backward_propagation(self, e, alpha):
        e = np.dot(self.activation_function.f_prime(self.z), e)
        de_dx = None

        return de_dx

    def conf2d(self, matrix, kernel):
        output_size = self.output_size[1] - self.kernel_size + 1
        output = np.empty([3, 3])
        print(output)
        offset = kernel // 2
        for row in range(3):
            for col in range(3):
                sub_matrix = matrix[row:row + 3, col:col + 3]
                output[row, col] = np.sum(sub_matrix * kernel)

        return output
