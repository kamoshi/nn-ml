import numpy as np


def zeros(input_size, output_size):
    return np.zeros(shape=(output_size, input_size))


def gaussian(scale: float = 1.0):
    def _gaussian(input_size, output_size):
        return np.random.normal(scale=scale, size=(output_size, input_size))
    return _gaussian


def xavier(input_size, output_size):
    return np.random.normal(scale=np.sqrt(2.0 / (input_size + output_size)), size=(output_size, input_size))


def he(input_size, output_size):
    return np.random.normal(scale=np.sqrt(2.0 / input_size), size=(output_size, input_size))
