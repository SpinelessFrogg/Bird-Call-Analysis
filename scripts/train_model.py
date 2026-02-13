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
    specs, labels = load_spectrogram_batches()

    builder = DatasetBuilder(specs, labels)

    X, y = builder.prepare(augment=True)
    spec_train, spec_test, labels_train, labels_test = builder.split(X, y)
    model = create_model(spec_train.shape[1:], labels_train.shape[1])
 
    # ### CHECKING DATA ###
    # import numpy as np
    # import matplotlib.pyplot as plt
    # print(np.unique(labels_train.argmax(axis=1), return_counts=True))
    # plt.imshow(spec_train[0].squeeze())
    # plt.colorbar()
    # plt.show()

    train_model(model, spec_train, spec_test, labels_train, labels_test)
    np.save(f"{MODEL_DIR}class_names.npy", builder.encoder.classes_)
    model.save(f"{MODEL_DIR}bird_model.keras")

if __name__ == "__main__":

    main()