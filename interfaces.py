import abc

from numpy.typing import NDArray


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

    @abc.abstractmethod
    def learn_weights(self, w_change: NDArray):
        ...

    @abc.abstractmethod
    def learn_bias(self, b_change: NDArray):
        ...
