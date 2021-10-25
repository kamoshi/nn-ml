import abc
from typing import Callable
import numpy as np


# Warstwy
class Layer(abc.ABC):
    @abc.abstractmethod
    def initialize(self, prev_size: int):
        ...

    @abc.abstractmethod
    def forward(self, inputs):
        ...

    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def weights(self):
        ...

    @property
    @abc.abstractmethod
    def bias(self):
        ...


class Input(Layer):
    def __init__(self, size: int):
        self._size = size

    def initialize(self, prev_size: int):
        pass

    def forward(self, inputs):
        return None, inputs

    @property
    def size(self):
        return self._size

    @property
    def weights(self):
        return None

    @property
    def bias(self):
        return None

    def __repr__(self):
        return f"Input({self._size})"


class Dense(Layer):
    def __init__(self, size: int, activation: Callable, w_init: Callable):
        self._size = size
        self._activation = activation
        self._bias = np.zeros(size)
        self._w_init = w_init
        self._weights = None

    def initialize(self, prev_size: int):
        self._weights = self._w_init(prev_size, self._size)

    def forward(self, inputs):
        z = self._weights.dot(inputs) + self._bias
        return z, self._activation(z)

    @property
    def size(self):
        return self._size

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    def __repr__(self):
        return f"Dense({self._size}, {self._activation.__name__})"


class NeuralNetwork:
    def __init__(self, input_size: int):
        self._input_layer = Input(input_size)
        self._layers: list[Layer] = [self._input_layer]

    def add_layer(self, layer: Layer):
        layer.initialize(self._layers[-1].size)
        self._layers.append(layer)

    def forward(self, inputs):
        outputs = []
        for layer in self._layers:
            z, inputs = layer.forward(inputs)
            outputs.append((z, inputs))
        return outputs

    def single_train(self, inputs, expected):
        outputs = self.forward(inputs)
        z, a = outputs[-1]
        grad = -(expected - a)
        print(grad)

    def __repr__(self):
        return f"NeuralNetwork({str(self._layers)})"
