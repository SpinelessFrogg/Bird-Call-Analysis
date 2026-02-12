from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocessing.pipeline import prepare_batch, add_noise
import numpy as np
class DatasetBuilder:
    def __init__(self, specs, labels, target_width=216):
        self.encoder = LabelEncoder()
        self.specs = specs
        self.labels = labels
        self.target_width = target_width

    def prepare(self, augment=False):
        X = prepare_batch(self.specs)
        if augment:
            X = add_noise()
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
    
        