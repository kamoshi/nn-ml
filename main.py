import numpy as np

from activation import sigmoid, softmax, relu, tanh
from initializers import gaussian
from layers import NeuralNetwork, Dense


if __name__ == '__main__':
    nn = NeuralNetwork(input_size=2)
    nn.add_layer(Dense(size=82, activation=tanh, w_init=gaussian()))
    nn.add_layer(Dense(size=82, activation=tanh, w_init=gaussian()))
    nn.add_layer(Dense(size=4, activation=softmax, w_init=gaussian()))
    print(nn)
    print(nn.forward(np.array([0, 0.2])))
