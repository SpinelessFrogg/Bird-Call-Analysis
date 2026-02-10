from tensorflow import keras
from keras import layers

# Neural network to detect patterns in the spectrogram visually
def detect_patterns(filters, block_no):
    pattern_layers = []
    for block in range(block_no):
        pool = (block < block_no - 1)
        pattern_layers.extend(conv_block(filters, pool))
        filters *= 2
    return pattern_layers

def conv_block(filters, pool=True):
    # checks for patterns in sample size (e.g. 3x3) in levels of filters (16, 32, 64) to detect simple and complex patterns
    block = [
        layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
    ]
    if pool:
        block.append(layers.MaxPooling2D((2, 2)))
    return block
    
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        *detect_patterns(32, 3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # model info
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model