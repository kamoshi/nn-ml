import functools

import numpy as np
from multiprocessing import Pool, RawArray

import numpy.random

from lab02.logic import optimizers
from lab02.logic.activation import softmax, activations
from lab02.logic.initializers import gaussian
from lab02.logic.layers.dense import Dense
from lab02.logic.network import NeuralNetwork
from lab02.utils.mnist_reader import load_mnist


def to_binary_output(y: int) -> np.ndarray:
    vec = np.zeros(10)
    vec[y] = 1
    return vec


var_dict = {}


# ładowanie danych procesu
def init_worker(
        x_train,
        x_train_shape,
        x_validate,
        x_validate_shape,
        x_test,
        x_test_shape,
        y_train,
        y_train_shape,
        y_validate,
        y_validate_shape,
        y_test,
        y_test_shape
):
    var_dict["x_train"] = x_train
    var_dict["x_train_shape"] = x_train_shape
    var_dict["x_validate"] = x_validate
    var_dict["x_validate_shape"] = x_validate_shape
    var_dict["x_test"] = x_test
    var_dict["x_test_shape"] = x_test_shape
    var_dict["y_train"] = y_train
    var_dict["y_train_shape"] = y_train_shape
    var_dict["y_validate"] = y_validate
    var_dict["y_validate_shape"] = y_validate_shape
    var_dict["y_test"] = y_test
    var_dict["y_test_shape"] = y_test_shape


# funkcja obliczająca wynik dla danego procesu
def worker_func(
        layer_activation,
        optimizer,
        worker_num: int,
):
    print(f"Worker {worker_num} started")
    x_train = np.frombuffer(var_dict["x_train"]).reshape(var_dict["x_train_shape"])
    x_validate = np.frombuffer(var_dict["x_validate"]).reshape(var_dict["x_validate_shape"])
    x_test = np.frombuffer(var_dict["x_test"]).reshape(var_dict["x_test_shape"])
    y_train = np.frombuffer(var_dict["y_train"]).reshape(var_dict["y_train_shape"])
    y_validate = np.frombuffer(var_dict["y_validate"]).reshape(var_dict["y_validate_shape"])
    y_test = np.frombuffer(var_dict["y_test"]).reshape(var_dict["y_test_shape"])

    numpy.random.seed(seed=worker_num)

    nn = NeuralNetwork(input_size=784)
    nn.add_layer(Dense(size=32, activation=activations[layer_activation], w_init=gaussian(scale=0.01)))
    nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))

    epochs = nn.sgd(
        x_train,
        y_train,
        max_epochs=1000,
        batch_size=50,
        learning_rate=0.1,
        stop_early=True,
        validate_data=(x_validate, y_validate),
        optimizer=optimizer
    )
    accuracy = nn.evaluate(x_test, y_test)
    print(f"Worker {worker_num} finished")
    return epochs, accuracy


# layer_size: int
# layer_activation: str
# initializer_scale: float
# batch_size: int
# learning_rate: float


# rozkręcenie procesora i sprawdzenie czy multiprocessing działa
test_00 = 0, [
    ('relu', optimizers.Default),
]

test_01 = 1, [
    ('sigmoid', optimizers.Momentum),
    ('sigmoid', optimizers.Nesterov),
    ('sigmoid', optimizers.Adagrad),
    ('sigmoid', optimizers.Adadelta),
    ('sigmoid', optimizers.Adam),
]

test_02 = 2, [
    ('relu', optimizers.Momentum),
    ('relu', optimizers.Nesterov),
    ('relu', optimizers.Adagrad),
    ('relu', optimizers.Adadelta),
    ('relu', optimizers.Adam),
]

test_cases = [test_00, test_01, test_02]


def main():
    x_train, y_train = load_mnist('data/mnist', kind='train')
    x_test, y_test = load_mnist('data/mnist', kind='t10k')
    y_train = np.array(list(map(to_binary_output, y_train)))
    y_test = np.array(list(map(to_binary_output, y_test)))

    x_train, x_validate = x_train[10000:], x_train[:10000]
    y_train, y_validate = y_train[10000:], y_train[:10000]

    # dzielone struktury danych
    x_train_raw = RawArray('d', x_train.shape[0] * x_train.shape[1])
    y_train_raw = RawArray('d', y_train.shape[0] * y_train.shape[1])
    x_validate_raw = RawArray('d', x_validate.shape[0] * x_validate.shape[1])
    y_validate_raw = RawArray('d', y_validate.shape[0] * y_validate.shape[1])
    x_test_raw = RawArray('d', x_test.shape[0] * x_test.shape[1])
    y_test_raw = RawArray('d', y_test.shape[0] * y_test.shape[1])

    # opakowanie danych w strukturę danych numpy array
    x_train_np = np.frombuffer(x_train_raw).reshape(x_train.shape)
    y_train_np = np.frombuffer(y_train_raw).reshape(y_train.shape)
    x_validate_np = np.frombuffer(x_validate_raw).reshape(x_validate.shape)
    y_validate_np = np.frombuffer(y_validate_raw).reshape(y_validate.shape)
    x_test_np = np.frombuffer(x_test_raw).reshape(x_test.shape)
    y_test_np = np.frombuffer(y_test_raw).reshape(y_test.shape)

    # kopiowanie danych do dzielonych struktur danych
    np.copyto(x_train_np, x_train)
    np.copyto(y_train_np, y_train)
    np.copyto(x_validate_np, x_validate)
    np.copyto(y_validate_np, y_validate)
    np.copyto(x_test_np, x_test)
    np.copyto(y_test_np, y_test)

    # tworzenie procesów
    processes = 10
    workers = 5
    with Pool(processes=processes, initializer=init_worker, initargs=(
            x_train_raw, x_train.shape,
            x_validate_raw, x_validate.shape,
            x_test_raw, x_test.shape,
            y_train_raw, y_train.shape,
            y_validate_raw, y_validate.shape,
            y_test_raw, y_test.shape,
    )) as pool:
        for i, test in test_cases:
            print("============\nRunning test", i)
            for activation, optimizer in test:
                print("With params:", activation, optimizer.__name__)

                _worker = functools.partial(worker_func, activation, optimizer)
                epochs, accuracies = zip(*pool.map(_worker, range(workers)))

                avg_epochs = sum(epochs) / len(epochs)
                avg_accuracy = sum(accuracies) / len(accuracies)

                print(f"Results (pool x{processes}):\n", epochs, accuracies)
                print("Avg epochs", avg_epochs, "Avg accuracy", avg_accuracy)

                # append result to file
                with open("lab02/results.csv", "a") as f:
                    f.write(f"{i},{activation},{optimizer.__name__},{avg_epochs},{avg_accuracy}\n")


if __name__ == '__main__':
    main()
    print("Done")
