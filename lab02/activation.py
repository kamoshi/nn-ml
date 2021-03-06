import numpy as np


def sigmoid():
    def _sigmoid(vec):
        return 1 / (1 + np.exp(-vec))

    def _sigmoid_p(a):
        return a * (1 - a)

    return _sigmoid, _sigmoid_p


def tanh():
    def _tanh(vec):
        return np.tanh(vec)

    def _tanh_p(a):
        return 1 - (a ** 2)

    return _tanh, _tanh_p


def relu():
    def _relu(vec):
        return np.maximum(0, vec)

    def _relu_p(a):
        return (a > 0).astype(a.dtype)

    return _relu, _relu_p


def softmax():
    def _softmax(vec):
        return (_exp := np.exp(vec)) / np.sum(_exp)

    return _softmax, None


activations = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax,
}
