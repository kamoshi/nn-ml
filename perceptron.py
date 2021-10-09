import random
from typing import Tuple, NewType


LabelledPoint = NewType("LabelledPoint", Tuple[list[int], int])


_FACTS: list[LabelledPoint] = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

def noisify_point(point: LabelledPoint) -> LabelledPoint:
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
    sigma = weighted_sum(X, W)
    activation = step_heaviside(sigma, theta=0)
    return activation


# Błąd
def error(d: int, y: int) -> int:
    return d - y


def simple_learning(X, Y, W, alpha) -> list[float]:
    get_error = lambda x, y, w: error(y=perceptron(x, w), d=y)

    def epoch() -> list[int]:
        errors = []
        for x, y in zip(X, Y):
            err = get_error(x, y, W)

            if err != 0:
                W[0] += alpha * err
                W[1] += alpha * err * x[0]
                W[2] += alpha * err * x[1]
            
            errors.append(err)
        return errors

    while True:
        print("Weights", W)
        print("Errors", errors := epoch())
        if all(item == 0 for item in errors):
            break
    
    return W


def test(X: list[list[float]], Y: list[int], W: list[float]):
    print("Testing weights:", W)
    print("Test size:", len(X))
    [print("Result:", result := perceptron(x, W), "\tExpected:", y, "\tDiff:", error(d=y, y=result)) for x, y in zip(X, Y)]


def main():
    X, Y = zip(*[noisify_point(_FACTS[random.randint(0, 3)]) for _ in range(1000)])
    X_train, X_test = X[200:], X[:200]
    Y_train, Y_test = Y[200:], Y[:200]
    W = [random.random()/10, random.random()/10, random.random()/10]
    alpha = 0.05
    test(X_test, Y_test, simple_learning(X_train, Y_train, W, alpha))


main()
