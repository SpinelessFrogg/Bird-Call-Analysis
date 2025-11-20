from tensorflow import keras
from keras import layers
from __main__ import native_birds


# Neural network to detect patterns in the spectrogram visually
def detect_patterns(filters, block_no, sample_size, shrink):
    pattern_layers = []
    # checks for patterns in specific sample size (e.g. 3x3) in levels of filters (16, 32, 64) to detect simple and complex patterns
    for block in range(block_no):
        pattern_layers.extend([
            layers.Conv2D(filters, (sample_size, sample_size), activation='relu', padding='same'),
            layers.MaxPooling2D((shrink, shrink))
        ])
        filters *= 2
    return pattern_layers

def create_model(spec_train, spec_test):
    input_shape = spec_train.shape[1:]
    num_classes = spec_test.shape[1]

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        detect_patterns(16, 3, 3, 2),

        # turn 2D data into 1D data for the computer; combine patterns to look at; ignore some nodes to prevent latching onto noise; give the final verdict
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    # model info
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model