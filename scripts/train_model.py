import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder
from preprocessing.pipeline import runtime_augment
from training.model import create_model
from training.training import train_model
from config import MODEL_DIR
import numpy as np

def main():
    X, y = load_spectrogram_batches()

    builder = DatasetBuilder(X, y)
    X_train_raw, X_test_raw, y_train, y_test = builder.split(X, y)
    builder.X = X_train_raw
    builder.y = y_train
    X_train, y_train = builder.prepare()

    builder.X = X_test_raw
    builder.y = y_test
    X_test, y_test = builder.prepare()
    np.save(f"{MODEL_DIR}X_test.npy", X_test)
    np.save(f"{MODEL_DIR}y_test.npy", y_test)

    train_data = runtime_augment(X_train, y_train, augment=True)
    test_data   = runtime_augment(X_test,  y_test,  augment=False)

    model = create_model(X_train.shape[1:], y_train.shape[1])

    train_model(model, train_data, test_data, y_train)
    np.save(f"{MODEL_DIR}class_names.npy", [cls.replace('_batch', '') for cls in builder.encoder.classes_])
    model.save(f"{MODEL_DIR}bird_model.keras")

if __name__ == "__main__":
    main()