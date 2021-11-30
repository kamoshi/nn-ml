import pickle
from collections import deque
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from lab02.logic import optimizers
from lab02.logic.interfaces import Layer
from lab02.logic.layers.input import Input


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

            nabla_b = delta = np.multiply(np.dot(forward_layer.w.T, delta.T).T, current_layer.derive())
            nabla_w = np.dot(delta.T, backward_layer.a)
            output.append((nabla_b, nabla_w))

        return output

    def sgd(self,
            inputs: NDArray,
            expected: list[NDArray],
            learning_rate: float,
            max_epochs: int,
            batch_size: int,
            stop_early: bool = False,
            validate_data: Tuple[list[NDArray], list[NDArray]] = None,
            optimizer=optimizers.Default,
            optimizer_kwargs: dict = None,
            ):
        optimizer = optimizer(self, **(optimizer_kwargs or {}))
        best_accuracy, best_model = 0.0, None
        last_scores = deque((best_accuracy,), maxlen=7)
        expected = np.array(expected)
        for epoch in range(max_epochs):
            # losowanie kolejności danych
            p = np.random.permutation(len(inputs))
            inputs, expected = inputs[p], expected[p]
            # podział na mini-batch
            for i in range(0, len(inputs), batch_size):
                optimizer.run_update(
                    self,
                    inputs[i:i + batch_size],
                    expected[i:i + batch_size],
                )
            # print(f"Epoch {epoch + 1}/{max_epochs}")

            # Ewaluacja na zbiorze walidacyjnym
            if validate_data is not None:
                accuracy = self.evaluate(validate_data[0], validate_data[1])
                # print("Epoch:", epoch, "Accuracy:", accuracy, "best:", best_accuracy)
                last_scores.append(accuracy)

                if stop_early and best_accuracy not in last_scores:
                    # print("Stopping training")
                    self.model = best_model
                    return epoch + 1
                if accuracy > best_accuracy:
                    best_accuracy, best_model = accuracy, self.model
        self.model = best_model

    def evaluate(self, inputs: list[NDArray], expected: list[NDArray]):
        correct = 0
        for i, x in enumerate(inputs):
            net_output = self.feedforward(x)
            if np.argmax(net_output) == np.argmax(expected[i]):
                correct += 1
        return correct / len(inputs)

    @property
    def layers(self):
        return self._layers.copy()

    @property
    def model(self) -> Tuple[list[NDArray], list[NDArray]]:
        return [layer.b for layer in self._layers[1:]], [layer.w for layer in self._layers[1:]]

    @model.setter
    def model(self, model):
        all_biases, all_weights = model
        for layer, bias, weights in zip(self._layers[1:], all_biases, all_weights):
            layer.b = bias
            layer.w = weights

    def save_model(self, path: str):
        model = str(self), self.model
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            architecture, model = pickle.load(f)
        if str(self) != architecture:
            print(f"Wrong architecture: {architecture}, this net is {self}")
        else:
            self.model = model

    def __repr__(self):
        return f"NeuralNetwork({str(self._layers)})"
