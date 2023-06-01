class NeuralNetwork:
    def __init__(self, layers, loss):
        for i in range(len(layers)-1):
            assert layers[i].output_size == layers[i+1].input_size

        self.layers = layers
        self.loss = loss

    def predict(self, x):
        result = []

        for i in range(len(x)):
            output = x[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                e = self.loss.f_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    e = layer.backward_propagation(e, learning_rate)

                err += self.loss.f(y_train[j], output)

            print('epoch %d/%d   error=%f' % (i+1, epochs, err / samples))
