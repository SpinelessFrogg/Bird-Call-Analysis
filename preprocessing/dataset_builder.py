import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DatasetBuilder:
    def __init__(self, specs, labels, target_width=216):
        self.encoder = LabelEncoder()
        self.specs = specs
        self.labels = labels
        self.target_width = target_width

    def prepare(self):
        X = np.array([self._fix_width(spec) for spec in self.specs], dtype=np.float32)
        X = self._normalize(X)
        X += 0.01 * np.random.randn(*X.shape) # this is probably not the best way to introduce confusion
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
    
    def _normalize(self, batch):
        # Normalize each spectrogram
        mean = np.mean(batch)
        std = np.std(batch) + 1e-9
        return (batch - mean) / std
    
    def _fix_width(self, spec):
        w = spec.shape[1]
        if w > self.target_width:
            return spec[:, :self.target_width]
        elif w < self.target_width:
            return np.pad(
                spec,
                ((0, 0), (0, self.target_width - w)),
                mode="constant"
            )
        return spec
        