from .layer import Layer
import numpy as np


class ReshapeLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, None)

    def forward_propagation(self, x):
        return x.reshape(self.output_size[0], self.output_size[1], self.output_size[2])

    def backward_propagation(self, e, alpha):
        return e.reshape(self.input_size[0], self.input_size[1], self.input_size[2])
