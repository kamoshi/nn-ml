import pickle

from keras.datasets.mnist import load_data
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def test_simple(x_train, y_train, x_test, y_test):
    simple_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10)
    ])

    simple_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    simple_model.fit(x_train, y_train, epochs=20)
    return simple_model.history, simple_model.evaluate(x_test, y_test)


def test_convolution(x_train, y_train, x_test, y_test):
    conv_model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (2, 2), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    conv_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    conv_model.fit(x_train, y_train, epochs=20)
    return conv_model.history, conv_model.evaluate(x_test, y_test)


def test_pooling(x_train, y_train, x_test, y_test):
    conv_model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    conv_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    conv_model.fit(x_train, y_train, epochs=20)
    return conv_model.history, conv_model.evaluate(x_test, y_test)


if __name__ == '__main__':
    train, test = load_data()

    results_simple = []
    for _ in range(10):
        history, accuracy = test_simple(*train, *test)
        results_simple.append((history.history, accuracy))
    pickle.dump(results_simple, open('results_simple.pkl', 'wb'))

    results_conv = []
    for _ in range(10):
        history, accuracy = test_convolution(*train, *test)
        results_conv.append((history.history, accuracy))
    pickle.dump(results_conv, open('results_conv.pkl', 'wb'))

    results_pool = []
    for _ in range(10):
        history, accuracy = test_pooling(*train, *test)
        results_pool.append((history.history, accuracy))
    pickle.dump(results_pool, open('results_pool.pkl', 'wb'))

