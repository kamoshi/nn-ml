import random
import itertools
from typing import Tuple
from shared import noisify_point, FACTS_OR, get_noise, get_sign


# Funkcja sumująca
def weighted_sum(X: list[float], W: list[float]) -> float:
    return W[0] + sum(x*w for x, w in zip(X, W[1:]))


# Funkcje aktywacji
def step_heaviside(sum: float, theta: float) -> 0 | 1:
    return 1 if sum > theta else 0

def step_bipolar(sum: float, theta: float) -> -1 | 1:
    return 1 if sum > theta else -1


# Perceptron
def perceptron(X: list[float], W: list[float], theta) -> int:
    return step_heaviside(weighted_sum(X, W), theta)


# Błąd
def error(d: int, y: int) -> int:
    return d - y


# Funkcja zwraca znalezione wagi i liczbę epok
def simple_learning(X: list[list[float]], Y: list[int], W: list[float], alpha: float, theta: float = 0) -> Tuple[list[float], int]:
    get_error = lambda x, y, w: error(y=perceptron(x, w, theta), d=y)
    W = W.copy()
    epochs = 0

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
        epochs += 1
        if epoch() == 0:
            break
    
    return W, epochs


def test(X: list[list[float]], Y: list[int], W: list[float], theta: float = 0):
    print("Testing weights:", W)
    print("Test size:", len(X))
    [print("Result:", result := perceptron(x, W, theta), "\tExpected:", y, "\tDiff:", error(d=y, y=result), "\tX: ", x) for x, y in zip(X, Y)]


def main():
    X, Y = zip(*[noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(500)])
    # X, Y = zip(*convert_points_to_bipolar([noisify_point(_FACTS_OR[random.randint(0, 3)]) for _ in range(500)]))
    X_train, X_test = X[50:], X[:50]
    Y_train, Y_test = Y[50:], Y[:50]

    W = [get_noise(10)*get_sign(), get_noise(10)*get_sign(), get_noise(10)*get_sign()]
    alpha = 0.01

    weights, epochs = simple_learning(X_train, Y_train, W, alpha, theta=0)
    print("Epochs:", epochs, "\tWeights:", weights)
    test(X_test, Y_test, weights, theta=0)


if __name__ == "__main__":
    main()
