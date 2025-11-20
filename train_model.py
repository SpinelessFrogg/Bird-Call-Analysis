from preprocessing import load_data, preprocess
from build_model import create_model
import os

def train_model(model, spec_train, spec_test, labels_train, labels_test):

    model.fit(
        spec_train, labels_train,
        validation_data=(spec_test, labels_test),
        epochs=20,
        batch_size=16,
        validation_split=0.2
    )

    model.save("birdcall_classify_model.keras")

# def evaluate_model(model, spec_test, labels_test):
#     test_loss, test_acc = model.evaluate(spec_test, labels_test)
#     print(f"Test accuracy: {test_acc:.2f}\nTest loss: {test_loss:.2f}")