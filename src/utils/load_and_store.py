import pickle
import os


def load_nn(filename):
    filepath = os.path.join('resources', filename)
    with open(filepath, 'rb') as file:
        nn = pickle.load(file)

    return nn


def store_nn(nn, filename):
    filepath = os.path.join('resources', filename)
    with open(filepath, 'wb') as file:
        pickle.dump(nn, file)
