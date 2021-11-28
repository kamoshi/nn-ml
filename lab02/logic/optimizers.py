import numpy as np
from numpy.typing import NDArray

from lab02.logic.interfaces import Optimizer
from lab02.logic.network import NeuralNetwork


class Default(Optimizer):
    def __init__(self, nn: NeuralNetwork, alpha: float = 0.9, gamma: float = 0.9):
        self.alpha, self.gamma = alpha, gamma

    def run_update(self, nn: NeuralNetwork, inputs: NDArray, expected: NDArray, learning_rate: float):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            sum_nabla_b = [np.add(snb, bnb) for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [np.add(snw, bnw) for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_w(self.alpha * sum_nabla_w.pop() / len(inputs))
            layer.learn_b(self.alpha * sum_nabla_b.pop() / len(inputs))


class Momentum(Optimizer):
    def __init__(self, nn: NeuralNetwork, alpha: float = 0.9, gamma: float = 0.9):
        self.alpha, self.gamma = alpha, gamma
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn: NeuralNetwork, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            b_nabla_b = [self.gamma * pgb + self.alpha * nb for pgb, nb in zip(self.cache_grad_b, b_nabla_b)]
            b_nabla_w = [self.gamma * pgw + self.alpha * nw for pgw, nw in zip(self.cache_grad_w, b_nabla_w)]
            sum_nabla_b = [np.add(snb, bnb) for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [np.add(snw, bnw) for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Zapisywanie poprzedniej aktualizacji
        self.cache_grad_b = [snb / len(inputs) for snb in sum_nabla_b]
        self.cache_grad_w = [snw / len(inputs) for snw in sum_nabla_w]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_w(sum_nabla_w.pop() / len(inputs))
            layer.learn_b(sum_nabla_b.pop() / len(inputs))


class Nesterov(Optimizer):
    def __init__(self, nn: NeuralNetwork, alpha: float = 0.9, gamma: float = 0.9):
        self.alpha, self.gamma = alpha, gamma
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn: NeuralNetwork, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Zmiana modelu na model z poprzednim wyliczonym gradientem
        current_model = nn.model
        nn.model = (
            [cmb - self.gamma * pgb for cmb, pgb in zip(current_model[0], self.cache_grad_b[::-1])],
            [cwm - self.gamma * pgw for cwm, pgw in zip(current_model[1], self.cache_grad_w[::-1])],
        )

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            b_nabla_b = [self.gamma * pgb + self.alpha * nb for pgb, nb in zip(self.cache_grad_b, b_nabla_b)]
            b_nabla_w = [self.gamma * pgw + self.alpha * nw for pgw, nw in zip(self.cache_grad_w, b_nabla_w)]
            sum_nabla_b = [np.add(snb, bnb) for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [np.add(snw, bnw) for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Zapisywanie poprzedniej aktualizacji
        self.cache_grad_b = [snb / len(inputs) for snb in sum_nabla_b]
        self.cache_grad_w = [snw / len(inputs) for snw in sum_nabla_w]

        # Przywrócenie pierwotnego modelu
        nn.model = current_model

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_w(self.alpha * sum_nabla_w.pop() / len(inputs))
            layer.learn_b(self.alpha * sum_nabla_b.pop() / len(inputs))


class Adagrad:
    def __init__(self, nn: NeuralNetwork, alpha: float = 0.9, epsilon: float = 1e-8):
        self.alpha, self.epsilon = alpha, epsilon
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn: NeuralNetwork, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            sum_nabla_b = [snb + bnb for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [snw + bnw for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Zapisywanie poprzedniej aktualizacji
        self.cache_grad_b = [cgb + ((snb / len(inputs)) ** 2) for snb, cgb in zip(sum_nabla_b, self.cache_grad_b)]
        self.cache_grad_w = [cgw + ((snw / len(inputs)) ** 2) for snw, cgw in zip(sum_nabla_w, self.cache_grad_w)]

        # Aplikowanie cache
        update_b = [self.alpha * snb / np.sqrt(cgb + self.epsilon) for snb, cgb in zip(sum_nabla_b, self.cache_grad_b)]
        update_w = [self.alpha * snw / np.sqrt(cgw + self.epsilon) for snw, cgw in zip(sum_nabla_w, self.cache_grad_w)]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_b(update_b.pop() / len(inputs))
            layer.learn_w(update_w.pop() / len(inputs))


class Adadelta:
    def __init__(self, nn: NeuralNetwork, alpha: float = 0.9, gamma: float = 0.9, epsilon: float = 1e-8):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_b_p = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]
        self.cache_grad_w_p = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn: NeuralNetwork, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            sum_nabla_b = [snb + bnb for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [snw + bnw for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Zapisywanie poprzedniej aktualizacji
        self.cache_grad_b = [self.gamma * cgb + (1 - self.gamma) * (snb / len(inputs)) ** 2 for cgb, snb in zip(self.cache_grad_b, sum_nabla_b)]
        self.cache_grad_w = [self.gamma * cgw + (1 - self.gamma) * (snw / len(inputs)) ** 2 for cgw, snw in zip(self.cache_grad_w, sum_nabla_w)]

        rms_b = [np.sqrt(cgb + self.epsilon) for cgb in self.cache_grad_b]
        rms_w = [np.sqrt(cgw + self.epsilon) for cgw in self.cache_grad_w]

        d_theta_b = [self.alpha / rb * cgb for rb, cgb in zip(rms_b, self.cache_grad_b)]
        d_theta_w = [self.alpha / rw * cgw for rw, cgw in zip(rms_w, self.cache_grad_w)]
        self.cache_grad_b_p = [self.gamma * cgbp + (1 - self.gamma) * dtb ** 2 for cgbp, dtb in zip(self.cache_grad_b_p, d_theta_b)]
        self.cache_grad_w_p = [self.gamma * cgwp + (1 - self.gamma) * dtw ** 2 for cgwp, dtw in zip(self.cache_grad_w_p, d_theta_w)]
        rms_theta_b = [np.sqrt(cgbp + self.epsilon) for cgbp in self.cache_grad_b_p]
        rms_theta_w = [np.sqrt(cgwp + self.epsilon) for cgwp in self.cache_grad_w_p]

        updates_b = [(rtb / rmsb * snb / len(inputs)) for rtb, rmsb, snb in zip(rms_theta_b, rms_b, sum_nabla_b)]
        updates_w = [(rtw / rmsw * snw / len(inputs)) for rtw, rmsw, snw in zip(rms_theta_w, rms_w, sum_nabla_w)]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_b(updates_b.pop())
            layer.learn_w(updates_w.pop())
