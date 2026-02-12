import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.metrics import PerformanceMetrics
from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder

def main():
    specs, labels = load_spectrogram_batches()
    builder = DatasetBuilder(specs, labels)
    X, y = builder.prepare()
    spec_train, spec_test, labels_train, labels_test = builder.split(X, y)
    model_performance = PerformanceMetrics(
        "2-10-26_fixedwidth_extra_conv.keras", 
        spec_test=spec_test, labels_test=labels_test)
    model_performance.evaluate_model()
    import numpy as np
    from config import MODEL_DIR
    np.save(f"{MODEL_DIR}class_names.npy", builder.encoder.classes_)
if __name__ == "__main__":
    main()