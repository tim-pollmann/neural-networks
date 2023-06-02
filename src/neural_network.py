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

    def fit(self, X_train, y_train, epochs, learning_rate):
        samples = len(X_train)

        for i in range(epochs):
            error_cum = 0
            if i == 250:
                learning_rate = learning_rate / 10

            if i == 1250:
                learning_rate = learning_rate / 10

            for j in range(samples):
                output = X_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                error_cum += self.loss.f(y_train[j], output)

                e = self.loss.f_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    e = layer.backward_propagation(e, learning_rate)

            error_epoch = error_cum / samples
            print(f'epoch {i+1}/{epochs}   error={error_epoch}')
