from random import random, randint
from typing import Tuple


_FACTS = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

def noisify_point(x, y) -> Tuple[int, int]:
    pass

_NOISE = [(random()/4)]


# Funkcja sumująca
def weighted_sum(X: list[float], W: list[float]) -> float:
    if len(X) != len(W):
        raise Exception("Input and weight vectors must have an equal length")
    sum = 0
    for i in range(0, len(X)):
        sum += X[i] * W[i]
    return sum


# Funkcje aktywacji
def step_heaviside(sum: float, theta: float) -> 0 | 1:
    return 1 if sum > theta else 0

def step_bipolar(sum: float, theta: float) -> -1 | 1:
    return 1 if sum > theta else -1


# Perceptron
def perceptron(X: list[float], W: list[float]) -> int:
    sum = W[0]  # bias = 1 * w[0]
    sum += weighted_sum(X, W[1:])  # w[1], w[2], ...
    activation = step_heaviside(sum, theta=0)
    return activation


def main():
    for (X, d) in _FACTS:
        result = perceptron(X, [-.4, 0.2, 0.3])
        print(f"X={X}, d={d}, res={result}")

main()
