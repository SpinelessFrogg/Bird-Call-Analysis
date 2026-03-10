from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.callbacks import EarlyStopping

from config import MODEL_DIR

def train_model(model, train_data, test_data, y_train):
     model.fit(
        train_data,
        validation_data=(test_data),
        epochs=100,
        batch_size=64,
        class_weight = weight_classes(y_train),
        callbacks=EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    )

def weight_classes(y_train):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train.argmax(axis=1)),
        y=y_train.argmax(axis=1)
    )
    class_weights = dict(enumerate(class_weights))
    return class_weights