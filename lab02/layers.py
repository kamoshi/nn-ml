import pickle
from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from interfaces import Layer


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
    def weights(self):
        raise ValueError("weights are not defined for Input layer")

    @weights.setter
    def weights(self, weights: NDArray):
        raise Exception("Input layer doesn't have weights")

    @property
    def bias(self):
        raise ValueError("bias is not defined for Input layer")

    @bias.setter
    def bias(self, bias: NDArray):
        raise Exception("Input layer doesn't have bias")

    def learn_weights(self, w_change: NDArray):
        raise Exception("Input layer cannot learn weights")

    def learn_bias(self, b_change: NDArray):
        raise Exception("Input layer cannot learn bias")

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
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: NDArray):
        assert(value.shape == self._weights.shape)
        self._weights = value.copy()

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value: NDArray):
        assert(value.shape == self._bias.shape)
        self._bias = value.copy()

    def learn_weights(self, w_change: NDArray):
        self._weights -= w_change

    def learn_bias(self, b_change: NDArray):
        self._bias -= b_change

    def __repr__(self):
        return f"Dense({self._size}, {self._activation.__name__})"


class NeuralNetwork:
    def __init__(self, input_size: int):
        self._input_layer = Input(input_size)
        self._layers: list[Layer] = [self._input_layer]

    def add_layer(self, layer: Layer):
        layer.initialize(self._layers[-1].size)
        self._layers.append(layer)

    def feedforward(self, inputs):
        for layer in self._layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, inputs: NDArray, expected: NDArray) -> list[Tuple[NDArray, NDArray]]:
        output: list[Tuple[NDArray, NDArray]] = []
        net_output = self.feedforward(inputs)

        # Liczenie błędu na wyjściu
        nabla_b = delta = -(expected - net_output)  # na podstawie funkcji Softmax
        nabla_w = np.dot(delta.T, self._layers[-2].a)
        output.append((nabla_b, nabla_w))

        # Liczenie błędu w warstwach ukrytych
        for i in range(len(self._layers) - 2):
            s_l, e_l = -i-3, -i
            backward_layer, current_layer, forward_layer = self._layers[s_l:None if e_l == 0 else e_l]

            nabla_b = delta = np.multiply(np.dot(forward_layer.weights.T, delta.T).T, current_layer.derive())
            nabla_w = np.dot(delta.T, backward_layer.a)
            output.append((nabla_b, nabla_w))

        return output

    def run_mini_batch(self, inputs: list[NDArray], expected: list[NDArray], learning_rate: float):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.bias.shape) for layer in self._layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.weights.shape) for layer in self._layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*self.backpropagate(x, y))
            sum_nabla_b = [np.add(snb, bnb) for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [np.add(snw, bnw) for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Aktualizacja wag i biasów
        for layer in self._layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_weights(learning_rate * sum_nabla_w.pop() / len(inputs))
            layer.learn_bias(learning_rate * sum_nabla_b.pop() / len(inputs))

    def sgd(self,
            inputs: list[NDArray],
            expected: list[NDArray],
            learning_rate: float,
            max_epochs: int,
            batch_size: int,
            stop_early: bool = False,
            test_data: Tuple[list[NDArray], list[NDArray]] = None
            ):
        last_accuracy = 0.0
        for epoch in range(max_epochs):
            for i in range(0, len(inputs), batch_size):
                self.run_mini_batch(inputs[i:i + batch_size], expected[i:i + batch_size], learning_rate)
            print(f"Epoch {epoch + 1}/{max_epochs}")
            # Ewaluacja na zbiorze testowym
            if test_data is not None:
                accuracy = self.evaluate(test_data[0], test_data[1])
                print("Accuracy:", accuracy)
                if stop_early and accuracy <= last_accuracy:
                    print("Overfitting, stopping training")
                    return epoch + 1
                last_accuracy = accuracy

    def evaluate(self, inputs: list[NDArray], expected: list[NDArray]):
        correct = 0
        for i, x in enumerate(inputs):
            net_output = self.feedforward(x)
            if np.argmax(net_output) == np.argmax(expected[i]):
                correct += 1
        return correct / len(inputs)

    def save_model(self, path: str):
        model = str(self), [layer.bias for layer in self._layers[1:]], [layer.weights for layer in self._layers[1:]]
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            model_str, biases, weights = pickle.load(f)
        print(f"Loaded model's net architecture: {model_str}")
        for layer, bias, weights in zip(self._layers[1:], biases, weights):
            layer.bias, layer.weights = bias, weights

    def __repr__(self):
        return f"NeuralNetwork({str(self._layers)})"
