import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.training import evaluate_model
from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder

def main():
    specs, labels = load_spectrogram_batches()
    builder = DatasetBuilder(specs, labels)
    X, y = builder.prepare()
    spec_train, spec_test, labels_train, labels_test = builder.split(X, y)
    evaluate_model(spec_test, labels_test)

if __name__ == "__main__":
    main()