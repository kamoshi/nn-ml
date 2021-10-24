import numpy as np


# Funkcje aktywacji
def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))


def tanh(vec):
    return np.tanh(vec)


def relu(vec):
    return np.maximum(0, vec)


def softmax(vec):
    return (_exp := np.exp(vec)) / np.sum(_exp)