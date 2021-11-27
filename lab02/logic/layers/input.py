import numpy as np
from numpy.typing import NDArray

from lab02.logic.interfaces import Layer


class Input(Layer):
    def __init__(self, size: int):
        self._size = size
        self._a = None

    def initialize(self, prev_size: int) -> None:
        raise Exception("Input layer cannot be initialized")

    def forward(self, inputs):
        assert(len(inputs) == self._size)
        # normalizacja wejścia
        if (max_value := np.max(inputs)) > 1:
            inputs = inputs / max_value
        self._a = inputs.reshape((1, len(inputs)))
        return self._a

    def derive(self):
        raise NotImplementedError

    @property
    def z(self):
        raise ValueError("z is not defined for Input layer")

    @property
    def a(self):
        return self._a

    @property
    def size(self):
        return self._size

    @property
    def w(self):
        raise ValueError("weights are not defined for Input layer")

    @w.setter
    def w(self, weights: NDArray):
        raise Exception("Input layer doesn't have weights")

    @property
    def b(self):
        raise ValueError("bias is not defined for Input layer")

    @b.setter
    def b(self, bias: NDArray):
        raise Exception("Input layer doesn't have bias")

    def learn_w(self, w_change: NDArray):
        raise Exception("Input layer cannot learn weights")

    def learn_b(self, b_change: NDArray):
        raise Exception("Input layer cannot learn bias")

    def __repr__(self):
        return f"Input({self._size})"
