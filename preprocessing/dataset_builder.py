import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def pad_to_median_width(batch_file):
    widths = [spec.shape[1] for spec in batch_file]
    target_w = int(np.median(widths))
    target_h = batch_file[0].shape[0]   # mel bins always same

    padded = []
    for spec in batch_file:
        h, w = spec.shape

        # Trim or pad width
        if w > target_w:
            spec = spec[:, :target_w]
        else:
            spec = np.pad(spec, ((0,0), (0, target_w - w)), mode="constant")

        padded.append(spec)

    return np.array(padded, dtype=np.float32)


def load_data():
    data_dir = "batch_data"
    batch_spectrogram_list = []
    species_names = []

    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            species = file.split("_batch.npy")[0]
            spectrograms = np.load(os.path.join(data_dir, file), allow_pickle=True)

            for spectro in spectrograms:
                spectro = np.array(spectro, dtype=np.float32)
                batch_spectrogram_list.append(spectro)
                species_names.append(species)

    batch_spectrogram_list = np.array(batch_spectrogram_list)
    species_names = np.array(species_names)
    return batch_spectrogram_list, species_names


def preprocess(batch_file, species_names):

    # Pad or trim to median width
    batch_file = pad_to_median_width(batch_file)

    # Normalize each spectrogram
    batch_file = np.array([
        (spec - spec.min()) / (spec.max() - spec.min() + 1e-9)
        for spec in batch_file
    ], dtype=np.float32)

    # Add channel dimension
    batch_file = np.expand_dims(batch_file, axis=-1)

    # Encode labels
    encoder = LabelEncoder()
    species_encoded = encoder.fit_transform(species_names)
    species_onehot = to_categorical(species_encoded)

    print("batch_file shape:", batch_file.shape)
    print("species_onehot shape:", species_onehot.shape)

    spec_train, spec_test, labels_train, labels_test = train_test_split(
        batch_file, species_onehot,
        test_size=0.2, random_state=42, stratify=species_onehot
    )

    return spec_train, spec_test, labels_train, labels_test, encoder
