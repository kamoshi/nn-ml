import random
import itertools
from typing import Tuple, NewType


LabelledPoint = NewType("LabelledPoint", Tuple[list[int], int])


_FACTS_AND: list[LabelledPoint] = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

_FACTS_OR: list[LabelledPoint] = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]


get_noise = lambda x: random.random() / x
get_sign = lambda: [-1, 1][random.randint(0, 1)]


def noisify_point(point: LabelledPoint) -> LabelledPoint:
    (x, y), c = point
    m_x, m_y, s_x, s_y = get_noise(4), get_noise(4), get_sign(), get_sign()
    return ([x + s_x*m_x, y + s_y*m_y], c)


# Funkcja sumująca
def weighted_sum(X: list[float], W: list[float]) -> float:
    return W[0] + sum(x*w for x, w in zip(X, W[1:]))


# Funkcje aktywacji
def step_heaviside(sum: float, theta: float) -> 0 | 1:
    return 1 if sum > theta else 0

def step_bipolar(sum: float, theta: float) -> -1 | 1:
    return 1 if sum > theta else -1


# Perceptron
def perceptron(X: list[float], W: list[float]) -> int:
    return step_heaviside(weighted_sum(X, W), theta=0)


# Błąd
def error(d: int, y: int) -> int:
    return d - y


def simple_learning(X: list[list[float]], Y: list[int], W: list[float], alpha: float) -> list[float]:
    get_error = lambda x, y, w: error(y=perceptron(x, w), d=y)

    def epoch() -> int:
        errors = 0
        for x, y in zip(X, Y):
            if (err := get_error(x, y, W)) != 0:
                errors += 1

            x_with_bias = itertools.chain((1.0,), x)
            for i in range(len(W)):
                W[i] += alpha * err * next(x_with_bias)
        return errors

    while True:
        print("Weights", W)
        print("Errors #", errors := epoch())
        if errors == 0:
            break
    
    return W


def test(X: list[list[float]], Y: list[int], W: list[float]):
    print("Testing weights:", W)
    print("Test size:", len(X))
    [print("Result:", result := perceptron(x, W), "\tExpected:", y, "\tDiff:", error(d=y, y=result)) for x, y in zip(X, Y)]


def main():
    X, Y = zip(*[noisify_point(_FACTS_OR[random.randint(0, 3)]) for _ in range(500)])
    X_train, X_test = X[50:], X[:50]
    Y_train, Y_test = Y[50:], Y[:50]

    W = [get_noise(10), get_noise(10), get_noise(10)]
    alpha = 0.01

    test(X_test, Y_test, simple_learning(X_train, Y_train, W, alpha))


if __name__ == "__main__":
    main()
