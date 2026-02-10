from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.callbacks import EarlyStopping

def train_model(model, spec_train, labels_train):
     model.fit(
        spec_train, labels_train,
        validation_split=0.1,
        epochs=20,
        batch_size=16,
        class_weight = weight_classes(labels_train),
        callbacks=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    )

def weight_classes(labels_train):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_train.argmax(axis=1)),
        y=labels_train.argmax(axis=1)
    )
    class_weights = dict(enumerate(class_weights))
    return class_weights

def evaluate_model(spec_test, labels_test, model):
    # model = load_model("birdcall_classify_model.keras")
    test_loss, test_accuracy = model.evaluate(spec_test, labels_test)
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Test loss: {test_loss:.3f}")