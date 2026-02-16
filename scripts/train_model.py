import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder

from training.model import create_model
from training.training import train_model
from config import MODEL_DIR
import numpy as np

def main():
    X, y = load_spectrogram_batches()

    builder = DatasetBuilder(X, y)

    X, y = builder.prepare(augment=True)
    X_train, X_test, y_train, y_test = builder.split(X, y)
    np.save(f"{MODEL_DIR}X_test.npy", X_test)
    np.save(f"{MODEL_DIR}y_test.npy", y_test)
    model = create_model(X_train.shape[1:], y_train.shape[1])
 
    # ### CHECKING DATA ###
    # import numpy as np
    # import matplotlib.pyplot as plt
    # print(np.unique(labels_train.argmax(axis=1), return_counts=True))
    # plt.imshow(spec_train[0].squeeze())
    # plt.colorbar()
    # plt.show()

    train_model(model, X_train, X_test, y_train, y_test)
    np.save(f"{MODEL_DIR}class_names.npy", [cls.replace('_batch', '') for cls in builder.encoder.classes_])
    model.save(f"{MODEL_DIR}bird_model.keras")

if __name__ == "__main__":
    main()