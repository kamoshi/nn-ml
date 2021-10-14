import random
from adaline import lms_learning
from shared import FACTS_OR, convert_points_to_bipolar, noisify_point, get_noise, get_sign


gen_sequences = lambda l: zip(*convert_points_to_bipolar([noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(l)]))
def gen_weights(l: int = 3, div: int = 10): return [get_noise(div) for _ in range(l)]


def test_weights():
    print("Running weight test.")
    w_ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    allowed_err = 0.4
    mi = 0.0001
    for w_range in w_ranges:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = list(map(lambda w: w * w_range, gen_weights(div=1)))
            results.append(lms_learning(X, Y, W, allowed_err, mi, max_epochs=100)[1])
        avg_epochs = sum(results)/len(results)
        print("Weight range:", w_range, "Epochs", avg_epochs)


def test_mi():
    mis = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    allowed_err = 0.4
    for mi in mis:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = gen_weights(div=1)
            results.append(lms_learning(X, Y, W, allowed_err, mi, max_epochs=100)[1])
        avg_epochs = sum(results) / len(results)
        print("Mi:", mi, "Epochs", avg_epochs)


def test_allowed_err():
    allowed_errs = [1.0, 0.8, 0.4, 0.3, 0.25]
    mi = 0.005
    for allowed_err in allowed_errs:
        results = []
        for _ in range(10):
            X, Y = gen_sequences(500)
            W = gen_weights(div=1)
            results.append(lms_learning(X, Y, W, allowed_err, mi)[1])
        avg_epochs = sum(results) / len(results)
        print("Allowed error:", allowed_err, "Epochs:", avg_epochs)



def main():
    test_weights()
    # test_mi()
    # test_allowed_err()
    pass

if __name__ == "__main__":
    main()
