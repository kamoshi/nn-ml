import numpy as np
from numpy.typing import NDArray

from lab02.logic import optimizers
from lab02.logic.activation import softmax, sigmoid
from lab02.logic.initializers import gaussian
from lab02.logic.layers.dense import Dense
from lab02.logic.network import NeuralNetwork
from lab02.utils.mnist_reader import load_mnist


def to_binary_output(y: int) -> NDArray:
    vec = np.zeros(10)
    vec[y] = 1
    return vec


def main():
    x_train, y_train = load_mnist('../data/mnist', kind='train')
    x_test, y_test = load_mnist('../data/mnist', kind='t10k')
    y_train = list(map(to_binary_output, y_train))
    y_test = list(map(to_binary_output, y_test))

    x_train, x_validate = x_train[10000:], x_train[:10000]
    y_train, y_validate = y_train[10000:], y_train[:10000]

    # x_train, y_train = gen_sequences(50000, FACTS_XOR)
    # x_test, y_test = gen_sequences(10000, FACTS_XOR)

    nn = NeuralNetwork(input_size=784)
    nn.add_layer(Dense(size=16, activation=sigmoid, w_init=gaussian(scale=0.01)))
    nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))

    nn.sgd(
        x_train,
        y_train,
        max_epochs=1,
        batch_size=50,
        learning_rate=0.1,
        stop_early=True,
        validate_data=(x_validate, y_validate),
        optimizer=optimizers.Adadelta
    )

    print("Test score evaluation:", nn.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()
