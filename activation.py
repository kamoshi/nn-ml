import numpy as np


def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))


def sigmoid_p(vec):
    return sigmoid(1 - sigmoid(vec))


def tanh(vec):
    return np.tanh(vec)


def tanh_p(vec):
    return 1 - (tanh(vec) ** 2)


def relu(vec):
    return np.maximum(0, vec)


def relu_p(vec):
    return np.where(vec > 0, 1, 0)


def softmax(vec):
    return (_exp := np.exp(vec)) / np.sum(_exp)
