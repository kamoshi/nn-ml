import numpy as np
from numpy.typing import NDArray

from activation import sigmoid, softmax, relu, tanh
from initializers import gaussian
from layers import NeuralNetwork, Dense
from utils.mnist_reader import load_mnist


def to_binary_output(y: int) -> NDArray:
    vec = np.zeros(10)
    vec[y] = 1
    return vec


def main():
    x_train, y_train = load_mnist('data/mnist', kind='train')
    x_test, y_test = load_mnist('data/mnist', kind='t10k')
    y_train = list(map(to_binary_output, y_train))
    y_test = list(map(to_binary_output, y_test))

    nn = NeuralNetwork(input_size=784)
    nn.add_layer(Dense(size=64, activation=sigmoid, w_init=gaussian()))
    nn.add_layer(Dense(size=64, activation=sigmoid, w_init=gaussian()))
    nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))

    nn.sgd(x_train, y_train, epochs=10, batch_size=50, learning_rate=0.05, test_data=(x_test, y_test))
    print(nn.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()
