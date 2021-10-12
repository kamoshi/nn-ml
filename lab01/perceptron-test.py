import random
from perceptron import simple_learning
from shared import FACTS_OR, noisify_point, get_noise, get_sign


def test_theta(X: list[list[float]], Y: list[float], W: list[float], alpha: float) -> float:
    print("Running theta test.")
    thetas = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    for theta in thetas:
        _, epochs = simple_learning(X, Y, W, alpha, theta)
        print("Theta:", theta, "\tEpochs:", epochs)


def test_weights(X, Y):
    print("Running weight test.")
    weight = [
        [1.0*get_sign(), 1.0*get_sign(), 1.0*get_sign()],
        [0.8*get_sign(), 0.8*get_sign(), 0.8*get_sign()],
        [0.6*get_sign(), 0.6*get_sign(), 0.6*get_sign()],
        [0.4*get_sign(), 0.4*get_sign(), 0.4*get_sign()],
        [0.2*get_sign(), 0.2*get_sign(), 0.2*get_sign()],
        [0.0*get_sign(), 0.0*get_sign(), 0.0*get_sign()],
    ]
    for weight in weight:
        _, epochs = simple_learning(X, Y, weight, alpha=0.05, theta=0)
        print("Epochs:", epochs, "\tWeights:", weight)


def test_alpha(X, Y, W):
    print("Running alpha test.")
    alphas = [1.0, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    for alpha in alphas:
        _, epochs = simple_learning(X, Y, W, alpha, theta=0)
        print("Alpha:", alpha, "\tEpochs:", epochs)


def test_activation(X, Y, W):
    print("Running activation test.")
    pass


def main():
    X, Y = zip(*[noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(500)])
    # X, Y = zip(*convert_points_to_bipolar([noisify_point(_FACTS_OR[random.randint(0, 3)]) for _ in range(500)]))
    X_train, X_test = X[50:], X[:50]
    Y_train, Y_test = Y[50:], Y[:50]

    W = [get_noise(10), get_noise(10), get_noise(10)]
    alpha = 0.005

    test_theta(X_train, Y_train, W, alpha)
    test_weights(X_train, Y_train)
    test_alpha(X_train, Y_train, W)
    test_activation(X_train, Y_train, W)


if __name__ == "__main__":
    main()
