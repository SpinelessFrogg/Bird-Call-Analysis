from keras.models import load_model
from config import MODEL_DIR
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class PerformanceMetrics:
    def __init__(self, model_name, spec_test, labels_test, builder):
        self.model = load_model(f"{MODEL_DIR}{model_name}")
        self.spec_test = spec_test
        self.labels_test = labels_test
        self.builder = builder

    def evaluate_model(self):
        self._loss_accuracy()
        self._conf_matrix()
        self._classif_report()
        for i, name in enumerate(self.builder.encoder.classes_):
            print(i, name)

    def _loss_accuracy(self):
        test_loss, test_accuracy = self.model.evaluate(self.spec_test, self.labels_test)
        print(f"Test accuracy: {test_accuracy:.3f}")
        print(f"Test loss: {test_loss:.3f}")

    def _conf_matrix(self):
        labels_pred = self.model.predict(self.spec_test)
        labels_pred = np.argmax(labels_pred, axis=1)
        labels_true = np.argmax(self.labels_test, axis=1)
        cm = confusion_matrix(labels_true, labels_pred)
        print(f"Confusion Matrix:\n{cm}")

    def _classif_report(self):
        labels_pred = self.model.predict(self.spec_test)
        labels_pred = np.argmax(labels_pred, axis=1)
        labels_true = np.argmax(self.labels_test, axis=1)
        cr = classification_report(labels_true, labels_pred)
        print(f"Classification Report:\n{cr}")

        ### Notes for me & class
        # Accuracy is good generally but not good for the big picture, see class 0 & 6, underrepresented and underperforming
        # Recall = how good at guessing positives, number of correct positives / number of actual positives (per class)
        # Precision = how many "yes" guesses were right, correct positives / all guessed positives (per class).