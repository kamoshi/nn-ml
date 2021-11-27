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
    def w(self):
        ...

    @w.setter
    @abc.abstractmethod
    def w(self, weights):
        ...

    @property
    @abc.abstractmethod
    def b(self):
        ...

    @b.setter
    @abc.abstractmethod
    def b(self, bias):
        ...

    @abc.abstractmethod
    def learn_w(self, w_change: NDArray):
        ...

    @abc.abstractmethod
    def learn_b(self, b_change: NDArray):
        ...
