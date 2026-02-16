import numpy as np
from pathlib import Path

def load_spectrogram_batches(dir="data/batches/"):
        X = []
        y = []

        dir = Path(dir)
        for file in dir.glob("*_batch.npy"): # gets every directory file (object) ending in _batch
            species = file.stem.split("_batch.npy")[0] # yoinks species name from file name 
            batch = np.load(file, allow_pickle=True)
            X.extend(batch) # adds all spectrograms to specs
            y.extend([species] * len(batch)) # adds labels equal to number of spectrograms in file.

        return np.array(X), np.array(y)