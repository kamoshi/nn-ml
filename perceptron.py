import random
from typing import Tuple, NewType


Entry = Tuple[list[int], int]


_FACTS: list[Entry] = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

def noisify_point(point: Entry) -> Entry:
    (x, y), c = point
    m_x, m_y = random.random() / 2, random.random() / 2
    s_x, s_y = [0, 1][random.randint(0, 1)], [0, 1][random.randint(0, 1)]
    return ([x + s_x*m_x, y + s_y*m_y], c)


# Funkcja sumująca
def weighted_sum(X: list[float], W: list[float]) -> float:
    s, W = W[0], W[1:]
    for i in range(0, len(X)):
        s += X[i] * W[i]
    return s


# Funkcje aktywacji
def step_heaviside(sum: float, theta: float) -> 0 | 1:
    return 1 if sum > theta else 0

def step_bipolar(sum: float, theta: float) -> -1 | 1:
    return 1 if sum > theta else -1


# Perceptron
def perceptron(X: list[float], W: list[float]) -> int:
    sum = weighted_sum(X, W)
    activation = step_heaviside(sum, theta=0)
    return activation


def error(d: int, y: int) -> int:
    return d - y


def simple_learning():
    W = [random.random()/4, random.random()/4, random.random()/4]
    data = [noisify_point(_FACTS[random.randint(0, 3)]) for _ in range(100)]

    X, Y = [t for t, _ in data], [c for _, c in data]
    X_train, X_test = X[20:], X[:20]
    Y_train, Y_test = Y[20:], Y[:20]



def main():
    simple_learning()

main()
