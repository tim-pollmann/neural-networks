from train_xor import train_xor
from train_digits import train_digits
from layers.conv_layer import ConvLayer
import numpy as np


def main():
    # train_digits()
    cl = ConvLayer((1, 5, 5), (1, 2, 3), 1, 3, None)
    m = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]
    ])
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    print(cl.conf2d(m, kernel))


if __name__ == '__main__':
    main()
