from keras.datasets.mnist import load_data
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    print("==== W pełni połączone warstwy ====")
    simple_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10)
    ])

    simple_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    simple_model.fit(x_train, y_train, epochs=5)
    simple_model.evaluate(x_test, y_test)

    print("==== Warstwy konwolucyjne ====")
    conv_model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (2, 2), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    conv_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    conv_model.fit(x_train, y_train, epochs=5)
    conv_model.evaluate(x_test, y_test)

    print("==== Warstwy konwolucyjne + MaxPooling ====")
    conv_model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    conv_model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    conv_model.fit(x_train, y_train, epochs=5)
    conv_model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
