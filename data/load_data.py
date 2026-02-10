import os
import numpy as np
from pathlib import Path

def load_spectrogram_batches(dir="data/batches/"):
        specs = []
        labels = []

        dir = Path(dir)
        for file in dir.glob("*_batch.npy"): # gets every directory file (object) ending in _batch
            species = file.stem.split("_batch.npy")[0] # yoinks species name from file name 
            batch = np.load(file, allow_pickle=True)

            specs.extend(batch) # adds all spectrograms to specs
            labels.extend([species] * len(batch)) # adds labels equal to number of spectrograms in file.

        return np.array(specs), np.array(labels)

            # for spectro in batch:
            #     spectro = np.array(spectro, dtype=np.float32)
            #     batch_spectrogram_list.append(spectro)
            #     species_names.append(species)

        # batch_spectrogram_list = np.array(batch_spectrogram_list)
        # species_names = np.array(species_names)
        # return batch_spectrogram_list, species_names
