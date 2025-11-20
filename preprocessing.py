import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data():
    data_dir = "batch_data"
    batch_spectrogram_list = [] 
    species_names = [] 

    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            species = file.split("_batch.npy")[0]
            spectrograms = np.load(os.path.join(data_dir, file), allow_pickle=True)

            for spectro in spectrograms:
                batch_spectrogram_list.append(spectro)
                species_names.append(species)

    batch_spectrogram_list = np.array(batch_spectrogram_list)
    species_names = np.array(species_names)
    return batch_spectrogram_list, species_names

def preprocess(batch_file, species_names):
    max_height = max(spectrogram.shape[0] for spectrogram in batch_file)
    max_width = max(spectrogram.shape[1] for spectrogram in batch_file)

    batch_file = np.array([np.pad(spectrogram, ((0, max_height - spectrogram.shape[0]), (0, max_width - spectrogram.shape[1]))) for spectrogram in batch_file])

    batch_file = (batch_file - batch_file.min() / (batch_file.max() - batch_file.min()))

    batch_file = np.expand_dims(batch_file, axis=-1)

    encoder = LabelEncoder()
    species_encoded = encoder.fit_transform(species_names)
    species_onehot = to_categorical(species_encoded)

    spec_train, spec_test, labels_train, labels_test = train_test_split(batch_file, species_onehot, test_size=0.2, random_state=42)
    return spec_train, spec_test, labels_train, labels_test, encoder
