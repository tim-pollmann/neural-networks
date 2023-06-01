import numpy as np

from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from activation_functions.softmax import Softmax
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.mse import MSE


def train_xor():
    x_train = np.array([
        [[0], [0], [0]],
        [[0], [0], [1]],
        [[0], [1], [0]],
        [[0], [1], [1]],
        [[1], [0], [0]],
        [[1], [0], [1]],
        [[1], [1], [0]],
        [[1], [1], [1]]
    ])

    y_train = np.array([
        [[1], [0], [0], [0]],
        [[0], [1], [0], [0]],
        [[0], [1], [0], [0]],
        [[0], [0], [1], [0]],
        [[0], [1], [0], [0]],
        [[0], [0], [1], [0]],
        [[0], [0], [1], [0]],
        [[0], [0], [0], [1]]
    ])

    nn = NeuralNetwork(
        [
            FCLayer(3, 10, Linear()),
            FCLayer(10, 10, Tanh()),
            FCLayer(10, 6, Tanh()),
            FCLayer(6, 4, Softmax())
        ],
        CrossEntropyLoss()
    )
    nn.fit(x_train, y_train, epochs=100, learning_rate=0.1)

    out = nn.predict(x_train)
    print(out)
