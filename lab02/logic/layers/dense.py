from typing import Callable

import numpy as np
from numpy.typing import NDArray

from lab02.logic.interfaces import Layer


class Dense(Layer):
    def __init__(self, size: int, activation: Callable, w_init: Callable):
        self._size = size
        self._activation, self._derivative = activation()
        self._bias = np.zeros((1, size))
        self._w_init = w_init
        self._weights = None
        self._z = None
        self._a = None

    def initialize(self, prev_size: int):
        self._weights = self._w_init(prev_size, self._size)

    def forward(self, inputs):
        self._z = np.dot(self._weights, inputs.T).T + self._bias
        self._a = self._activation(self._z)
        return self.a

    def derive(self):
        return self._derivative(self.a)

    @property
    def z(self):
        return self._z

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size

    @property
    def w(self):
        return self._weights

    @w.setter
    def w(self, value: NDArray):
        assert(value.shape == self._weights.shape)
        self._weights = value.copy()

    @property
    def b(self):
        return self._bias

    @b.setter
    def b(self, value: NDArray):
        assert(value.shape == self._bias.shape)
        self._bias = value.copy()

    def learn_w(self, w_change: NDArray):
        self._weights -= w_change

    def learn_b(self, b_change: NDArray):
        self._bias -= b_change

    def __repr__(self):
        return f"Dense({self._size}, {self._activation.__name__})"
