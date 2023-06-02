import numpy as np
import sklearn.datasets


def load_digits_1d():
    x, y = sklearn.datasets.load_digits(return_X_y=True)
    x_preprocessed = np.array([x.reshape(-1, 1)[np.newaxis] for x in x])
    y_preprocessed = preprocess_y(y)
    return x_preprocessed, y_preprocessed


def load_digits_2d():
    x, y = sklearn.datasets.load_digits(return_X_y=True)
    x_preprocessed = np.array([x.reshape(8, 8)[np.newaxis] for x in x])
    y_preprocessed = preprocess_y(y)
    return x_preprocessed, y_preprocessed

def preprocess_y(y):
    y_preprocessed = np.empty([len(y), 1, 10, 1], dtype=float)
    for i, y in enumerate(y):
        y_ohe = np.zeros([10, 1], dtype=float)
        y_ohe[y] = 1.
        y_preprocessed[i] = y_ohe[np.newaxis]
    return y_preprocessed