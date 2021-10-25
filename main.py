import numpy as np

from activation import sigmoid, softmax, relu, tanh
from initializers import gaussian
from layers import NeuralNetwork, Dense
from utils.mnist_reader import load_mnist


def main():
    X_train, y_train = load_mnist('data/mnist', kind='train')
    X_test, y_test = load_mnist('data/mnist', kind='t10k')

    nn = NeuralNetwork(input_size=3)
    nn.add_layer(Dense(size=4, activation=tanh, w_init=gaussian()))
    nn.add_layer(Dense(size=2, activation=softmax, w_init=gaussian()))

    nn.single_train(np.array([0, 0.2, 0.4]), np.array([0, 0]))


if __name__ == '__main__':
    main()
