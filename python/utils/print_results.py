import numpy as np


def print_ohe(d, y):
    y = np.argmax(y, axis=2)
    d = np.argmax(d, axis=2)

    for i in range(len(y)):
        print(f'pred={y[i, 0, 0]}, true={d[i, 0, 0]}')

