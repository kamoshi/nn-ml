import numpy as np

from activation import sigmoid, softmax, relu, tanh
from initializers import gaussian
from layers import NeuralNetwork, Dense


if __name__ == '__main__':
    nn = NeuralNetwork(input_size=2)
    nn.add_layer(Dense(size=3, activation=sigmoid, w_init=gaussian()))
    # nn.add_layer(Dense(size=32, activation=relu, w_init=gaussian()))
    # nn.add_layer(Dense(size=10, activation=softmax, w_init=gaussian()))

    print(nn.forward(np.array([0, 0])))
