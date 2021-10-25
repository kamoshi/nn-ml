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

    @abc.abstractmethod
    def derive(self):
        ...

    @property
    @abc.abstractmethod
    def z(self):
        ...

    @property
    @abc.abstractmethod
    def a(self):
        ...

    @property
    @abc.abstractmethod
    def size(self):
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
        self._a = None

    def initialize(self, prev_size: int):
        pass

    def forward(self, inputs):
        assert(len(inputs) == self._size)
        self._a = inputs.reshape((1, len(inputs)))
        return self._a

    def derive(self):
        return None

    @property
    def z(self):
        return None

    @property
    def a(self):
        return self._a

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
        self._activation, self._derivative = activation()
        self._bias = np.zeros((1, size))
        self._w_init = w_init
        self._weights = None
        self._z = None
        self._a = None

    def initialize(self, prev_size: int):
        self._weights = self._w_init(prev_size, self._size)

    def forward(self, inputs):
        self._z = (self._weights @ inputs.T).T + self._bias
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
        for layer in self._layers:
            inputs = layer.forward(inputs)
        return inputs

    def single_train(self, inputs, expected):
        output = self.forward(inputs)
        d = (-(expected - output))
        gradient = d.T @ self._layers[-2].a

        for i in range(len(self._layers) - 2):
            previous_layer = self._layers[-3 - i]
            current_layer = self._layers[-2 - i]
            forward_layer = self._layers[-1 - i]
            d = np.multiply(forward_layer.weights.T.dot(d.T).T, current_layer.derive())
            gradient = d.T.dot(previous_layer.a)

    def __repr__(self):
        return f"NeuralNetwork({str(self._layers)})"
