import itertools
import random
from shared import FACTS_OR, noisify_point, convert_points_to_bipolar, get_sign


# Funkcja sumująca
def weighted_sum(X: list[float], W: list[float]) -> float:
    return W[0] + sum(x*w for x, w in zip(X, W[1:]))


# sgn(y)
def threshold(sum: float) -> -1 | 1:
    return 1 if sum > 0 else -1


# Błąd
def error(d: int, y: float) -> float:
    return d - y

def quadratic_error(d: int, y: int) -> int:
    return (d - y) ** 2


def adaline(X: list[float], W: list[float]) -> float:
    return threshold(weighted_sum(X, W))


def lms_learning(X: list[list[float]], Y: list[int], W: list[float], allowed_err: float, mi: float, max_epochs: int = 100) -> list[float]:
    epochs = 0
    def epoch() -> float:
        summed_quadratic_err = 0.0
        for x, y in zip(X, Y):
            result = weighted_sum(x, W)
            err = error(d=y, y=result)
            summed_quadratic_err += quadratic_error(d=y, y=result)

            x_with_bias = itertools.chain((1.0,), x)
            for i in range(len(W)):
                W[i] += 2 * mi * err * next(x_with_bias)
        
        return summed_quadratic_err / len(X)
    
    while True:
        epochs += 1
        mean_quadratic_err = epoch()
        max_epochs -= 1
        if mean_quadratic_err < allowed_err or max_epochs <= 0:
            break
    
    return W, epochs


def test(X: list[list[float]], Y: list[int], W: list[float]):
    print("Testing weights:", W)
    print("Test size:", len(X))
    [print("Result:", result := adaline(x, W), "\tExpected:", y, "\tDiff:", error(d=y, y=result), "\tX: ", x) for x, y in zip(X, Y)]


def main():
    X, Y = zip(*convert_points_to_bipolar([noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(500)]))
    X_train, X_test = X[50:], X[:50]
    Y_train, Y_test = Y[50:], Y[:50]
    W = [
        random.random()*get_sign(),
        random.random()*get_sign(),
        random.random()*get_sign(),
    ]

    weights = lms_learning(X_train, Y_train, W, allowed_err=0.7, mi=0.05, max_epochs=100)
    test(X_test, Y_test, weights)


if __name__ == "__main__":
    main()
