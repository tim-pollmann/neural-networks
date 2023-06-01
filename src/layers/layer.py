class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def forward_propagation(self, x):
        raise NotImplementedError

    def backward_propagation(self, e, alpha):
        raise NotImplementedError
