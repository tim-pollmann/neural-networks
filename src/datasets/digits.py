import numpy as np
import sklearn.datasets


def load_digits():
    x, y = sklearn.datasets.load_digits(return_X_y=True)
    x_preprocessed = [np.expand_dims(x.reshape(-1, 1), axis=0) for x in x]
    y_preprocessed = np.empty([len(y), 10, 1], dtype=float)
    for i, y in enumerate(y):
        y_ohe = np.zeros([10, 1], dtype=float)
        y_ohe[y] = 1.
        y_preprocessed[i] = y_ohe

    return x_preprocessed, y_preprocessed
