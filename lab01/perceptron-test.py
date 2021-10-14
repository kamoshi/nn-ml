import random
from perceptron import simple_learning, step_bipolar
from shared import FACTS_OR, noisify_point, get_noise, get_sign


gen_sequences = lambda l: zip(*[noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(l)])
def gen_weights(l: int = 3, div: int = 10): return [get_noise(div) for _ in range(l)]


def test_theta():
    print("Running theta test.")
    thetas = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    alpha = 0.005
    for theta in thetas:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = gen_weights()
            results.append(simple_learning(X, Y, W, alpha, theta)[1])
        avg_epochs = sum(results)/len(results)
        print("Theta:", theta, "\tAvg epochs:", avg_epochs)


def test_weights():
    print("Running weight test.")
    w_ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha = 0.005
    for w_range in w_ranges:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = list(map(lambda w: w * w_range, gen_weights(div=1)))
            results.append(simple_learning(X, Y, W, alpha)[1])
        avg_epochs = sum(results)/len(results)
        print("Weight range:", w_range, "Epochs", avg_epochs)


def test_alpha():
    print("Running alpha test.")
    alphas = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    for alpha in alphas:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = gen_weights()
            results.append(simple_learning(X, Y, W, alpha)[1])
        avg_epochs = sum(results)/len(results)
        print("Alpha:", alpha, "\tEpochs:", avg_epochs)


def test_activation():
    alphas = [0.1, 0.01, 0.001, 0.0001]
    for alpha in alphas:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(200)
            Y_b = [-1 if y == 0 else y for y in Y]
            W = gen_weights(div=1)
            results.append((simple_learning(X, Y, W, alpha)[1], simple_learning(X, Y_b, W, alpha, activation=step_bipolar)[1]))
        res_count = len(results)
        result = tuple(res / res_count for res in map(sum, zip(*results)))
        print("Alpha:", alpha, "Unipolar:", result[0], "Bipolar:", result[1])


def main():
    # test_theta()
    test_weights()
    # test_alpha()
    # test_activation()


if __name__ == "__main__":
    main()
