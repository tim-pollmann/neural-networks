from neural_network import NeuralNetwork
from layers.fc_layer import FCLayer
from activation_functions.tanh import Tanh
from activation_functions.linear import Linear
from activation_functions.softmax import Softmax
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.mse import MSE

from datasets.digits import load_digits

from utils.load_and_store import load_nn, store_nn
from utils.print_results import print_ohe
from utils.split_dataset import split_dataset


LOAD = False
STORE = False


def train_digits():
    x, y = load_digits()
    x_train, y_train, x_valid, y_valid = split_dataset(x, y)

    if LOAD:
        nn = load_nn('nn_digits.pkl')
    else:
        nn = NeuralNetwork(
            [
                FCLayer((1, 64, 1), (1, 256, 1), Linear()),
                FCLayer((1, 256, 1), (1, 256, 1), Tanh()),
                FCLayer((1, 256, 1), (1, 256, 1), Tanh()),
                FCLayer((1, 256, 1), (1, 128, 1), Tanh()),
                FCLayer((1, 128, 1), (1, 10, 1), Softmax())
            ],
            CrossEntropyLoss()
        )
        nn.fit(x_train, y_train, epochs=2000, learning_rate=0.00001)

    pred = nn.predict(x_valid)
    print_ohe(y_valid, pred)

    if STORE:
        store_nn(nn, 'nn_digits.pkl')
