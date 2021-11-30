import numpy as np
from numpy.typing import NDArray

from lab02.logic.interfaces import Optimizer


class Default(Optimizer):
    def __init__(self, nn, alpha: float = 0.9):
        self.alpha = alpha

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
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

    def __repr__(self):
        return f"Optimizer.Default({self.alpha})"


# noinspection DuplicatedCode
class Momentum(Optimizer):
    def __init__(self, nn, alpha: float = 0.1, gamma: float = 0.8):
        self.alpha, self.gamma = alpha, gamma
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
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

    def __repr__(self):
        return f"Optimizer.Momentum({self.alpha}, {self.gamma})"


# noinspection DuplicatedCode
class Nesterov(Optimizer):
    def __init__(self, nn, alpha: float = 0.1, gamma: float = 0.8):
        self.alpha, self.gamma = alpha, gamma
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
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

    def __repr__(self):
        return f"Optimizer.Nesterov({self.alpha}, {self.gamma})"


class Adagrad:
    def __init__(self, nn, alpha: float = 0.1, epsilon: float = 1e-8):
        self.alpha, self.epsilon = alpha, epsilon
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
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

    def __repr__(self):
        return f"Optimizer.Adagrad({self.alpha}, {self.epsilon})"


# noinspection DuplicatedCode
class Adadelta:
    def __init__(self, nn, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1e-8):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.cache_grad_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_b_p = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_grad_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]
        self.cache_grad_w_p = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            sum_nabla_b = [snb + bnb for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [snw + bnw for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        # Zapisywanie poprzedniej aktualizacji
        self.cache_grad_b = [self.gamma * cgb + (1 - self.gamma) * (snb / len(inputs)) ** 2 for cgb, snb in
                             zip(self.cache_grad_b, sum_nabla_b)]
        self.cache_grad_w = [self.gamma * cgw + (1 - self.gamma) * (snw / len(inputs)) ** 2 for cgw, snw in
                             zip(self.cache_grad_w, sum_nabla_w)]

        rms_b = [np.sqrt(cgb + self.epsilon) for cgb in self.cache_grad_b]
        rms_w = [np.sqrt(cgw + self.epsilon) for cgw in self.cache_grad_w]

        d_theta_b = [self.alpha / rb * cgb for rb, cgb in zip(rms_b, self.cache_grad_b)]
        d_theta_w = [self.alpha / rw * cgw for rw, cgw in zip(rms_w, self.cache_grad_w)]
        self.cache_grad_b_p = [self.gamma * cgbp + (1 - self.gamma) * dtb ** 2 for cgbp, dtb in
                               zip(self.cache_grad_b_p, d_theta_b)]
        self.cache_grad_w_p = [self.gamma * cgwp + (1 - self.gamma) * dtw ** 2 for cgwp, dtw in
                               zip(self.cache_grad_w_p, d_theta_w)]
        rms_theta_b = [np.sqrt(cgbp + self.epsilon) for cgbp in self.cache_grad_b_p]
        rms_theta_w = [np.sqrt(cgwp + self.epsilon) for cgwp in self.cache_grad_w_p]

        updates_b = [(rtb / rmsb * snb / len(inputs)) for rtb, rmsb, snb in zip(rms_theta_b, rms_b, sum_nabla_b)]
        updates_w = [(rtw / rmsw * snw / len(inputs)) for rtw, rmsw, snw in zip(rms_theta_w, rms_w, sum_nabla_w)]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_b(updates_b.pop())
            layer.learn_w(updates_w.pop())

    def __repr__(self):
        return f"Optimizer.Adadelta({self.alpha}, {self.gamma}, {self.epsilon})"


# noinspection DuplicatedCode
class Adam:
    def __init__(self, nn, alpha: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.alpha, self.beta1, self.beta2, self.epsilon = alpha, beta1, beta2, epsilon
        self.cache_m_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_v_b = [np.zeros(b.shape) for b in nn.model[0]][::-1]
        self.cache_m_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]
        self.cache_v_w = [np.zeros(b.shape) for b in nn.model[1]][::-1]

    def run_update(self, nn, inputs: NDArray, expected: NDArray):
        # Przechowywanie sumy nabli wag i sumy nabli biasów
        sum_nabla_b = [np.zeros(layer.b.shape) for layer in nn.layers[1:][::-1]]
        sum_nabla_w = [np.zeros(layer.w.shape) for layer in nn.layers[1:][::-1]]

        # Sumowanie wyników z propagacji wstecznej
        for x, y in zip(inputs, expected):
            b_nabla_b, b_nabla_w = zip(*nn.backpropagate(x, y))
            sum_nabla_b = [snb + bnb for snb, bnb in zip(sum_nabla_b, b_nabla_b)]
            sum_nabla_w = [snw + bnw for snw, bnw in zip(sum_nabla_w, b_nabla_w)]

        self.cache_m_b = [self.beta1 * cmb + (1 - self.beta1) * (snb / len(inputs)) for cmb, snb in
                          zip(self.cache_m_b, sum_nabla_b)]
        self.cache_v_b = [self.beta2 * cvb + (1 - self.beta2) * (snb / len(inputs)) ** 2 for cvb, snb in
                          zip(self.cache_v_b, sum_nabla_b)]
        self.cache_m_w = [self.beta1 * cmw + (1 - self.beta1) * (snw / len(inputs)) for cmw, snw in
                          zip(self.cache_m_w, sum_nabla_w)]
        self.cache_v_w = [self.beta2 * cvw + (1 - self.beta2) * (snw / len(inputs)) ** 2 for cvw, snw in
                          zip(self.cache_v_w, sum_nabla_w)]

        mt_hat_b = [mtb / (1 - self.beta1) for mtb in self.cache_m_b]
        vt_hat_b = [vtb / (1 - self.beta2) for vtb in self.cache_v_b]
        update_b = [self.alpha / (np.sqrt(vthat) + self.epsilon) * mthat for mthat, vthat in zip(mt_hat_b, vt_hat_b)]

        mt_hat_w = [mtw / (1 - self.beta1) for mtw in self.cache_m_w]
        vt_hat_w = [vtw / (1 - self.beta2) for vtw in self.cache_v_w]
        update_w = [self.alpha / (np.sqrt(vthat) + self.epsilon) * mthat for mthat, vthat in zip(mt_hat_w, vt_hat_w)]

        # Aktualizacja wag i biasów
        for layer in nn.layers[1:]:  # pomijamy warstwę wejściową
            layer.learn_b(update_b.pop() / len(inputs))
            layer.learn_w(update_w.pop() / len(inputs))

    def __repr__(self):
        return f"Optimizer.Adam({self.alpha}, {self.beta1}, {self.beta2}, {self.epsilon})"
