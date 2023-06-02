import numpy as np
import random


def split_dataset(x, y, ratio=0.8, shuffle=True):
    dataset = list(zip(x, y))
    random.shuffle(dataset)
    split = int(len(dataset) * ratio)
    x, y = zip(*dataset)
    return x[:split], y[:split], x[split:], y[split:]
