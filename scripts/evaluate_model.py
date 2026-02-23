import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.metrics import PerformanceMetrics
from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder

def main():
    X, y = load_spectrogram_batches()
    builder = DatasetBuilder(X, y)
    X, y = builder.prepare()
    X_train, X_test, y_train, y_test = builder.split(X, y)
    model_performance = PerformanceMetrics(
        model_name="2-20-26_waveform_aug.keras",
        # model_name="2-10-26_fixedwidth_extra_conv.keras",
        X_test=X_test, y_test=y_test, builder=builder)
    model_performance.evaluate_model()

if __name__ == "__main__":
    main()