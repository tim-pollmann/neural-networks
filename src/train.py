import numpy as np
from sklearn.datasets import load_digits

from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from activation_functions.softmax import Softmax
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.mse import MSE


# # x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
# # y_train = np.array([[[0]], [[1]], [[1]], [[2]]])

# X, y = load_digits(return_X_y=True)
# X_preprocessed = [x.reshape(-1,1) for x in X]
# y_train_preprocessed = np.empty([len(y), 10, 1], dtype=float)
# for i, y in enumerate(y):
#     y_ohe = np.zeros([10, 1], dtype=float)
#     y_ohe[y] = 1.
#     y_train_preprocessed[i] = y_ohe

# nn = NeuralNetwork(
#     [
#         FCLayer(64, 256, Tanh()),
#         FCLayer(256, 128, Tanh()),
#         FCLayer(128, 10, Linear())
#     ],
#     MSE()
# )
# nn.fit(X_preprocessed, y_train_preprocessed, epochs=1000, learning_rate=0.1)

# # out = nn.predict(x_train)
# # print(out)

x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
y_train = np.array([[[0], [1], [0]], [[1], [0], [0]], [[1], [0], [0]], [[1], [0], [0]]])

nn = NeuralNetwork(
    [
        FCLayer(2, 4, Tanh()),
        FCLayer(4, 3, Linear()),
        FCLayer(3, 3, Softmax())
    ],
    CrossEntropyLoss()
)
nn.fit(x_train, y_train, epochs=100, learning_rate=0.1)

out = nn.predict(x_train)
print(out)
