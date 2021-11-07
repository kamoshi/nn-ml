import numpy as np
from numpy.typing import NDArray

from lab02.utils.mnist_reader import load_mnist


def to_binary_output(y: int) -> NDArray:
    vec = np.zeros(10)
    vec[y] = 1
    return vec


def main():
    x_train, y_train = load_mnist('data/mnist', kind='train')
    x_test, y_test = load_mnist('data/mnist', kind='t10k')
    y_train = list(map(to_binary_output, y_train))
    y_test = list(map(to_binary_output, y_test))

    # nn = NeuralNetwork(input_size=784)
    # nn.add_layer(Dense(size=128, activation=relu, w_init=gaussian(scale=0.01)))
    # nn.add_layer(Dense(size=64, activation=relu, w_init=gaussian(scale=0.01)))
    # nn.add_layer(Dense(size=32, activation=tanh, w_init=gaussian(scale=0.01)))
    # nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))
    #
    # epochs = nn.sgd(x_train, y_train, max_epochs=20, batch_size=50, learning_rate=0.1, stop_early=True, test_data=(x_test, y_test))
    #
    # nn.save_model('model_data')
    # nn.load_model('model_data')
    #
    # print(nn.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()
