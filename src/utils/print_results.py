import numpy as np


def print_ohe(d, y):
    y = np.argmax(y, axis=1)
    d = np.argmax(d, axis=1)

    for i in range(len(y)):
        print(f'pred={y[i]}, true={d[i]}')
