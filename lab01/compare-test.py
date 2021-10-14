import random
from adaline import lms_learning
from perceptron import simple_learning, step_bipolar
from shared import FACTS_OR, convert_points_to_bipolar, noisify_point, get_noise, get_sign



def main():
    alphas_mis = [.1, .01, .001, .0001, .00001]
    for am in alphas_mis:
        results = []
        for _ in range(10):
            X, Y = zip(*convert_points_to_bipolar([noisify_point(FACTS_OR[random.randint(0, 3)]) for _ in range(500)]))
            X_train, X_test = X[50:], X[:50]
            Y_train, Y_test = Y[50:], Y[:50]
            W = [get_noise(1)*get_sign() for _ in range(3)]
            weights_p = simple_learning(X_train, Y_train, W, alpha=am, activation=step_bipolar)[1]
            weights_a = lms_learning(X_train, Y_train, W.copy(), allowed_err=0.3, mi=am, max_epochs=1200)[1]
            results.append((weights_p, weights_a))
        l = len(results)
        s = tuple(map(sum, zip(*results)))
        print("Alpha/Mi:", am, "Perceptron epochs:", s[0]/l, "ADALINE epochs:", s[1]/l)


if __name__ == "__main__":
    main()