import numpy as np


def zeros(input_size, output_size):
    return np.zeros((input_size, output_size))


def gaussian(scale: float = 1.0):
    def _gaussian(input_size, output_size):
        return np.random.normal(scale, size=(output_size, input_size))
    return _gaussian
