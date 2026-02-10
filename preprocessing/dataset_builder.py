import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DatasetBuilder:
    def __init__(self, specs, labels):
        self.encoder = LabelEncoder()
        self.specs = specs
        self.labels = labels

    def prepare(self):
        # X = self._pad_to_median_width(self.specs) # I think that padding is causing major issues with the data. 5s cropping already standardized?
        X = np.array(self.specs, dtype=np.float32)
        X = self._normalize(X)
        X += 0.01 * np.random.randn(*X.shape)
        # Add channel dimension
        X = np.expand_dims(X, axis=-1)

        species_encoded = self.encoder.fit_transform(self.labels)
        y = to_categorical(species_encoded)
        return X, y

    def split(self, X, y, test_size=0.2):
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42
        )

    def _pad_to_median_width(self, batch):
        if batch.size == 0:
            raise ValueError("Batch is empty")
        widths = [spec.shape[1] for spec in batch]
        if not widths:  # Check if widths is empty
            raise ValueError("No widths found in the batch.")
        target_w = int(np.median(widths))

        padded = []
        for spec in batch:
            h, w = spec.shape
            # Trim or pad width
            if w > target_w:
                spec = spec[:, :target_w]
            else:
                spec = np.pad(spec, ((0, 0), (0, target_w - w)), mode="constant")
            padded.append(spec)
        return np.array(padded, dtype=np.float32)

    def _normalize(self, batch):
        # Normalize each spectrogram
        mean = np.mean(batch)
        std = np.std(batch) + 1e-9
        return (batch - mean) / std
        