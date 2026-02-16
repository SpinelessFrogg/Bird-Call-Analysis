from keras.models import load_model
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn

class PerformanceMetrics:
    def __init__(self, model_name, X_test, y_test, builder):
        self.model = load_model(f"{MODEL_DIR}{model_name}")
        self.X_test = X_test
        self.y_test = y_test
        self.builder = builder
        self.class_names = np.load(f"{MODEL_DIR}class_names.npy")

    def evaluate_model(self):
        loss_acc = self._loss_accuracy()
        print(f"Test accuracy: {loss_acc[0]}")
        print(f"Test loss: {loss_acc[1]}")
        print(f"Confusion Matrix:\n{self._conf_matrix()}")
        print(f"Classification Report:\n{self._classif_report()}")
        for i, name in enumerate(self.class_names):
            print(i, name)
        self._conf_heatmap(cm=self._conf_matrix())

    def _loss_accuracy(self):
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        return test_accuracy, test_loss

    def _conf_matrix(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        return cm

    def _conf_heatmap(self, cm):
        plt.figure(figsize=(12,10))

        seaborn.heatmap(
            cm,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot=True,
            fmt="d",
            cmap="Reds"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.show()

    def _classif_report(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        cr = classification_report(y_true, y_pred)
        return cr

        ### Notes for me & class
        # Accuracy is good generally but not good for the big picture, see class 0 & 6, underrepresented and underperforming
        # Recall = how good at guessing positives, number of correct positives / number of actual positives (per class)
        # Precision = how many "yes" guesses were right, correct positives / all guessed positives (per class).