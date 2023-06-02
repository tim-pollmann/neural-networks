from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from layers.reshape_layer import ReshapeLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from activation_functions.softmax import Softmax
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.mse import MSE

from datasets.digits import load_digits_1d, load_digits_2d

from utils.load_and_store import load_nn, store_nn
from utils.print_results import print_ohe
from utils.split_dataset import split_dataset


LOAD = True
STORE = False


def train_digits():
    x, y = load_digits_2d()
    x_train, y_train, x_valid, y_valid = split_dataset(x, y)

    if LOAD:
        nn = load_nn('nn_digits.pkl')
    else:
        nn = NeuralNetwork(
            [
                ReshapeLayer((1, 8, 8), (1, 64, 1)),
                FCLayer((1, 64, 1), (1, 256, 1), Linear()),
                FCLayer((1, 256, 1), (1, 256, 1), Tanh()),
                FCLayer((1, 256, 1), (1, 256, 1), Tanh()),
                FCLayer((1, 256, 1), (1, 128, 1), Tanh()),
                FCLayer((1, 128, 1), (1, 10, 1), Softmax())
            ],
            CrossEntropyLoss((1, 10, 1))
        )
        nn.fit(x_train, y_train, epochs=200, learning_rate=0.00001)

    pred = nn.predict(x_valid)
    print_ohe(y_valid, pred)

    if STORE:
        store_nn(nn, 'nn_digits.pkl')
