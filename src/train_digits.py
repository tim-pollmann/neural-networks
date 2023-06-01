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


LOAD = True
STORE = False


def train_digits():
    x, y = load_digits()

    if LOAD:
        nn = load_nn('nn_digits.pkl')
    else:
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
        nn.fit(x[:-5], y[:-5], epochs=250, learning_rate=0.00001)

    pred = nn.predict(x[-5:])
    print_ohe(y[-5:], pred)

    if STORE:
        store_nn(nn, 'nn_digits.pkl')
