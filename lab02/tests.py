import functools

import numpy as np
from multiprocessing import Pool, RawArray

from activation import softmax, activations
from initializers import gaussian
from lab02.layers import NeuralNetwork, Dense
from utils.mnist_reader import load_mnist


def to_binary_output(y: int) -> np.ndarray:
    vec = np.zeros(10)
    vec[y] = 1
    return vec


var_dict = {}


# ładowanie danych procesu
def init_worker(x_train, x_train_shape, x_test, x_test_shape, y_train, y_train_shape, y_test, y_test_shape):
    var_dict["x_train"] = x_train
    var_dict["x_train_shape"] = x_train_shape
    var_dict["x_test"] = x_test
    var_dict["x_test_shape"] = x_test_shape
    var_dict["y_train"] = y_train
    var_dict["y_train_shape"] = y_train_shape
    var_dict["y_test"] = y_test
    var_dict["y_test_shape"] = y_test_shape


# funkcja obliczająca wynik dla danego procesu
def worker_func(
        layer_size: int,
        layer_activation: str,
        initializer_scale: float,
        batch_size: int,
        learning_rate: float,
        worker_num: int,
):
    print(f"Worker {worker_num} started")
    x_train = np.frombuffer(var_dict["x_train"]).reshape(var_dict["x_train_shape"])
    x_test = np.frombuffer(var_dict["x_test"]).reshape(var_dict["x_test_shape"])
    y_train = np.frombuffer(var_dict["y_train"]).reshape(var_dict["y_train_shape"])
    y_test = np.frombuffer(var_dict["y_test"]).reshape(var_dict["y_test_shape"])

    nn = NeuralNetwork(input_size=784)
    nn.add_layer(Dense(size=layer_size, activation=activations[layer_activation], w_init=gaussian(scale=initializer_scale)))
    nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))

    epochs = nn.sgd(x_train, y_train, max_epochs=100, batch_size=batch_size, learning_rate=learning_rate, stop_early=True, validate_data=(x_test, y_test))
    accuracy = nn.evaluate(x_test, y_test)
    print(f"Worker {worker_num} finished")
    return epochs, accuracy


# layer_size: int
# layer_activation: str
# initializer_scale: float
# batch_size: int
# learning_rate: float


# rozkręcenie procesora i sprawdzenie czy multiprocessing działa
test_00 = [
    (8, "relu", 0.1, 50, 0.1)
]


# liczba neuronów w warstwie ukrytej
test_01 = [
    (16, "sigmoid", 0.01, 50, 0.1),
    (32, "sigmoid", 0.01, 50, 0.1),
    (64, "sigmoid", 0.01, 50, 0.1),
    (128, "sigmoid", 0.01, 50, 0.1),
]

# wpływ współczynnika uczenia
test_02 = [
    (32, "sigmoid", 0.01, 50, 0.01),
    (32, "sigmoid", 0.01, 50, 0.1),
    (32, "sigmoid", 0.01, 50, 1),
    (32, "sigmoid", 0.01, 50, 10),
]

# wpływ rozmiaru mini-batcha
test_03 = [
    (32, "sigmoid", 0.01, 1, 0.1),
    (32, "sigmoid", 0.01, 10, 0.1),
    (32, "sigmoid", 0.01, 50, 0.1),
    (32, "sigmoid", 0.01, 100, 0.1),
]

# wpływ inicjalizacji wag
test_04 = [
    (32, "sigmoid", 10, 50, 0.1),
    (32, "sigmoid", 1, 50, 0.1),
    (32, "sigmoid", 0.1, 50, 0.1),
    (32, "sigmoid", 0.01, 50, 0.1),
]

# wpływ aktywacji
test_05 = [
    (32, "sigmoid", 0.01, 50, 0.01),
    (32, "relu", 0.01, 50, 0.01),
    (32, "tanh", 0.01, 50, 0.01),
]


test_cases = [
    test_00,
    test_01,
    test_02,
    test_03,
    test_04,
    test_05,
]


def main():
    x_train, y_train = load_mnist('data/mnist', kind='train')
    x_test, y_test = load_mnist('data/mnist', kind='t10k')
    y_train = np.array(list(map(to_binary_output, y_train)))
    y_test = np.array(list(map(to_binary_output, y_test)))

    # dzielone struktury danych
    x_train_raw = RawArray('d', x_train.shape[0] * x_train.shape[1])
    y_train_raw = RawArray('d', y_train.shape[0] * y_train.shape[1])
    x_test_raw = RawArray('d', x_test.shape[0] * x_test.shape[1])
    y_test_raw = RawArray('d', y_test.shape[0] * y_test.shape[1])

    # opakowanie danych w strukturę danych numpy array
    x_train_np = np.frombuffer(x_train_raw).reshape(x_train.shape)
    y_train_np = np.frombuffer(y_train_raw).reshape(y_train.shape)
    x_test_np = np.frombuffer(x_test_raw).reshape(x_test.shape)
    y_test_np = np.frombuffer(y_test_raw).reshape(y_test.shape)

    # kopiowanie danych do dzielonych struktur danych
    np.copyto(x_train_np, x_train)
    np.copyto(y_train_np, y_train)
    np.copyto(x_test_np, x_test)
    np.copyto(y_test_np, y_test)

    # tworzenie procesów
    processes = 10
    with Pool(processes=processes, initializer=init_worker, initargs=(x_train_raw, x_train.shape, x_test_raw, x_test.shape, y_train_raw, y_train.shape, y_test_raw, y_test.shape)) as pool:
        for i, test in enumerate(test_cases):
            print("============\nRunning test", i)
            for size, activation, scale, batch_size, learning_rate in test:
                print("With params:", size, activation, scale, batch_size, learning_rate)

                _worker = functools.partial(worker_func, size, activation, scale, batch_size, learning_rate)
                epochs, accuracies = zip(*pool.map(_worker, range(processes)))

                avg_epochs = sum(epochs) / len(epochs)
                avg_accuracy = sum(accuracies) / len(accuracies)

                print(f"Results (pool x{processes}):\n", epochs, accuracies)
                print("Avg epochs", avg_epochs, "Avg accuracy", avg_accuracy)

                # append result to file
                with open("results.txt", "a") as f:
                    f.write(f"{i},{size},{activation},{scale},{batch_size},{learning_rate},{avg_epochs},{avg_accuracy}\n")


if __name__ == '__main__':
    main()
    print("Done")
