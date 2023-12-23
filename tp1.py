import matplotlib.pyplot as plt
import numpy as np
from tp1_utils import load_data

import tensorflow as tf
from keras.models import Sequential
from keras import layers


def training_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def show_prediction(x_test, y_test, model, pokemon_types, multi_label):
    rand_num = np.random.randint(0, y_test.shape[0])
    image = x_test[rand_num]

    true_class = [pokemon_types[i] for i in np.where(y_test[rand_num] == 1)[0]]

    out = model(np.expand_dims(x_test[rand_num], axis=0)).numpy()[0]
    ind = np.argwhere(out.round() == 1).flatten() if multi_label else np.argmax(out)
    prediction = [pokemon_types[i] for i in ind] if multi_label else [pokemon_types[ind]]

    plt.tight_layout()
    plt.imshow(image)

    plt.title(f'True = {" & ".join(true_class)}\nPredicted = {" & ".join(prediction)}')

    plt.axis('off')
    plt.show()


def mlp(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, x_test, y_test, pokemon_types):
    print("""MLP model""")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=INPUT_SHAPE),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Hyperparameters

    EPOCHS = 50
    VAL_SPLIT = 0.125
    LEARNING_RATE = 0.2

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history_mlp = model.fit(x_train, y_train, validation_split=VAL_SPLIT, epochs=EPOCHS)

    training_plot(history_mlp)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print(f'Test acc={test_acc}\tTest loss={test_loss}')

    show_prediction(x_test, y_test, model, pokemon_types, multi_label=False)


def cnn(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, x_test, y_test, pokemon_types):
    print("""Convolutional Neural Network""")

    # Conv2d and MaxPooling layers

    model = Sequential([
        layers.Conv2D(256, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal(), input_shape=INPUT_SHAPE),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((2, 2))
    ])

    # Classifier layers

    classification_head = Sequential([
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.add(classification_head)

    # Hyperparameters

    EPOCHS = 20
    VAL_SPLIT = 0.125
    LEARNING_RATE = 0.001

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history_cnn = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=VAL_SPLIT)

    training_plot(history_cnn)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(f'Test acc={test_acc}\tTest loss={test_loss}')

    show_prediction(x_test, y_test, model, pokemon_types, multi_label=False)


def multi_label_cnn(INPUT_SHAPE, NUM_CLASSES, x_train, y_train_2types, x_test, y_test_2types, pokemon_types):
    print("""Multi-label Classification""")

    # Feature extraction layers

    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal(), input_shape=INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((3, 3)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same",
                      kernel_initializer=tf.keras.initializers.HeNormal()),
    ])

    # Classification head

    classification_head = Sequential([
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='sigmoid'),
    ])
    model.add(classification_head)

    EPOCHS = 20
    VAL_SPLIT = 0.125
    LEARNING_RATE = 1e-3

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history_cnn_multi = model.fit(x_train, y_train_2types, epochs=EPOCHS, validation_split=VAL_SPLIT)

    training_plot(history_cnn_multi)

    test_loss, test_acc = model.evaluate(x_test, y_test_2types, verbose=2)

    print(f'Test acc={test_acc}\tTest loss={test_loss}')

    show_prediction(x_test, y_test_2types, model, pokemon_types, multi_label=True)


def conv_next_xlarge(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types,
                     pokemon_types, single_label):
    print("""Pre-trained deep neural network models\nConvNetXtXLarge""")

    feature_extractor = tf.keras.applications.ConvNeXtXLarge(input_shape=INPUT_SHAPE, include_top=False,
                                                             include_preprocessing=False)
    feature_extractor.trainable = False

    # Hyperparameters

    EPOCHS = 20
    VAL_SPLIT = 0.125
    LEARNING_RATE = 0.001

    if single_label:
        print("""Single-label classification""")

        model = Sequential([
            feature_extractor,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=VAL_SPLIT)

        training_plot(history)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

        print(f'Test acc={test_acc}\tTest loss={test_loss}')

        show_prediction(x_test, y_test, model, pokemon_types, multi_label=False)

    else:
        print("""Multi-label Classification""")

        model = Sequential([
            feature_extractor,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        history_multi = model.fit(x_train, y_train_2types, epochs=EPOCHS, validation_split=VAL_SPLIT)

        training_plot(history_multi)

        test_loss, test_acc = model.evaluate(x_test, y_test_2types, verbose=2)

        print(f'Test acc={test_acc}\tTest loss={test_loss}')

        show_prediction(x_test, y_test_2types, model, pokemon_types, multi_label=True)


def vgg16(NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types, pokemon_types, single_label):
    print("""VGG16""")

    feature_extractor = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    feature_extractor.trainable = False

    # Hyperparameters

    EPOCHS = 20
    VAL_SPLIT = 0.125
    LEARNING_RATE = 0.001

    if single_label:
        print("""Single-label classification""")

        model = Sequential([
            layers.Resizing(height=224, width=224),
            feature_extractor,
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=VAL_SPLIT)

        training_plot(history)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

        print(f'Test acc={test_acc}\tTest loss={test_loss}')

        show_prediction(x_test, y_test, model, pokemon_types, multi_label=False)
    else:
        print("""Multi-label Classification""")

        model = Sequential([
            layers.Resizing(height=224, width=224),
            feature_extractor,
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        history_multi = model.fit(x_train, y_train_2types, epochs=EPOCHS, validation_split=VAL_SPLIT)

        training_plot(history_multi)

        test_loss, test_acc = model.evaluate(x_test, y_test_2types, verbose=2)

        print(f'Test acc={test_acc}\tTest loss={test_loss}')

        show_prediction(x_test, y_test_2types, model, pokemon_types, multi_label=True)


if __name__ == '__main__':
    print("""Load dataset and pokemon types""")

    dataset = load_data()
    print(type(dataset))
    print(dataset.keys())

    pokemon_types = []
    pokemon_types_file = open('dataset/pokemon_types.txt')
    for pokemon_type in pokemon_types_file:
        pokemon_types.append(pokemon_type.strip())
    pokemon_types_file.close()
    print(pokemon_types)

    print("""Check dataset information""")

    x_train = dataset['train_X']
    y_train = dataset['train_classes']
    y_train_2types = dataset['train_labels']

    x_test = dataset['test_X']
    y_test = dataset['test_classes']
    y_test_2types = dataset['test_labels']

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train_2types shape: {y_train_2types.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_test_2types shape: {y_test_2types.shape}")

    print(f"Train: {x_train.shape[0]}")
    print(f"Test: {x_test.shape[0]}")

    INPUT_SHAPE = x_train.shape[1:]
    NUM_CLASSES = y_train.shape[1]

    print("""Visualize pokemon images""")

    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    for i in range(16):
        rand_num = np.random.randint(0, y_train.shape[0])
        pokemon_img = plt.subplot(4, 4, i + 1)
        plt.imshow(x_train[rand_num])

        indices = np.where(y_train_2types[rand_num] == 1)[0]
        primary_type = indices[0]
        title = f'{pokemon_types[primary_type]} & {pokemon_types[indices[1]]}' if len(
            indices) > 1 else f'{pokemon_types[primary_type]}'
        plt.title(title)

        plt.axis('off')
    plt.show()

    experiment = input("""
    Choose an experiment
    A. MLP
    B. CNN
    C. Multi-label CNN
    D. Single-label ConveNeXtXLarge
    E. Multi-label ConveNeXtXLarge
    F. Single-label VGG16
    G. Multi-label VGG16
    """).lower()

    if experiment == 'a':
        mlp(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, x_test, y_test, pokemon_types)
    elif experiment == 'b':
        cnn(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, x_test, y_test, pokemon_types)
    elif experiment == 'c':
        multi_label_cnn(INPUT_SHAPE, NUM_CLASSES, x_train, y_train_2types, x_test, y_test_2types, pokemon_types)
    elif experiment == 'd':
        conv_next_xlarge(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types,
                         pokemon_types, single_label=True)
    elif experiment == 'e':
        conv_next_xlarge(INPUT_SHAPE, NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types,
                         pokemon_types, single_label=False)
    elif experiment == 'f':
        vgg16(NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types, pokemon_types,
              single_label=True)
    elif experiment == 'g':
        vgg16(NUM_CLASSES, x_train, y_train, y_train_2types, x_test, y_test, y_test_2types, pokemon_types,
              single_label=False)

    print('END')
