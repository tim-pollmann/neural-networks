import numpy as np
from sklearn.datasets import load_digits

from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from activation_functions.softmax import Softmax
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.mse import MSE


# x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[2]]])

X, y = load_digits(return_X_y=True)
x_preprocessed = [x.reshape(-1, 1) for x in X]
y_preprocessed = np.empty([len(y), 10, 1], dtype=float)
for i, y in enumerate(y):
    y_ohe = np.zeros([10, 1], dtype=float)
    y_ohe[y] = 1.
    y_preprocessed[i] = y_ohe


nn = NeuralNetwork(
    [
        FCLayer(64, 256, Linear()),
        FCLayer(256, 256, Tanh()),
        FCLayer(256, 256, Tanh()),
        FCLayer(256, 128, Tanh()),
        FCLayer(128, 10, Softmax())
    ],
    CrossEntropyLoss()
)
nn.fit(x_preprocessed[:-5], y_preprocessed[:-5], epochs=100, learning_rate=0.00001)

print(np.argmax(y_preprocessed[:5], axis=1))
print(np.argmax(nn.predict(x_preprocessed[:5]), axis=1))
print(np.argmax(y_preprocessed[-5:], axis=1))
print(np.argmax(nn.predict(x_preprocessed[-5:]), axis=1))
# def print_result(di, yi):
#     print(f'pred={yi.index(max(yi))}, true={di.index(max(di))}')

# for i in range(5):
#     print_result(y_preprocessed[i], y_preprocessed[i])


# x_train = np.array([
#     [[0], [0], [0]],
#     [[0], [0], [1]],
#     [[0], [1], [0]],
#     [[0], [1], [1]],
#     [[1], [0], [0]],
#     [[1], [0], [1]],
#     [[1], [1], [0]],
#     [[1], [1], [1]]
# ])
# y_train = np.array([
#     [[1], [0], [0], [0]],
#     [[0], [1], [0], [0]],
#     [[0], [1], [0], [0]],
#     [[0], [0], [1], [0]],
#     [[0], [1], [0], [0]],
#     [[0], [0], [1], [0]],
#     [[0], [0], [1], [0]],
#     [[0], [0], [0], [1]]
# ])

# nn = NeuralNetwork(
#     [
#         FCLayer(3, 10, Linear()),
#         FCLayer(10, 10, Tanh()),
#         FCLayer(10, 6, Tanh()),
#         FCLayer(6, 4, Softmax())
#     ],
#     CrossEntropyLoss()
# )
# nn.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# out = nn.predict(x_train)
# print(out)
