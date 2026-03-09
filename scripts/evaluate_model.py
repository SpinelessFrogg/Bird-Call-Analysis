import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR
from training.metrics import PerformanceMetrics
import numpy as np

def main():
    X_test = np.load(f"{MODEL_DIR}X_test.npy")
    y_test = np.load(f"{MODEL_DIR}y_test.npy")
    class_names = np.load(f"{MODEL_DIR}class_names.npy")
    model_performance = PerformanceMetrics(
        model_name="bird_model.keras",
        # model_name="2-10-26_fixedwidth_extra_conv.keras",
        X_test=X_test, y_test=y_test)
    model_performance.evaluate_model()

if __name__ == "__main__":
    main()