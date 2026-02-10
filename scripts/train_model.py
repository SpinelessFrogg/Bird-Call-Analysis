from data.load_data import load_spectrogram_batches
from preprocessing.dataset_builder import DatasetBuilder
from training.model import create_model
from training.training import train_model
from config import MODEL_DIR

def main():
    specs, labels = load_spectrogram_batches()

    builder = DatasetBuilder(specs, labels)

    X, y = builder.prepare()
    spec_train, spec_test, labels_train, labels_test = builder.split(X, y)

    model = create_model(spec_train.shape[1:], spec_test.shape[1])

    train_model(model, spec_train, labels_train)

    model.save(MODEL_DIR / "bird_model.keras")

if __name__ == "__main__":
    main()