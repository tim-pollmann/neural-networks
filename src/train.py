import numpy as np

from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from losses.mse import MSE


x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[2]]])


nn = NeuralNetwork(
    [
        FCLayer(2, 3, Tanh()),
        FCLayer(3, 1, Linear())
    ],
    MSE()
)
nn.fit(x_train, y_train, epochs=100, learning_rate=0.1)

out = nn.predict(x_train)
print(out)
