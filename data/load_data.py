import numpy as np
from pathlib import Path

def load_spectrogram_batches(dir="data/batches/", split="train"):
    X = []
    y = []

    dir = Path(dir)
    for file in dir.glob(f"*_{split}_batch.npy"): # gets every directory file (object) ending in _batch
        species = file.stem.replace(f"_{split}_batch", "") # yoinks species name from file name 
        batch = np.load(file, allow_pickle=True)

        for item in batch:
            X.append(item["spec"])
            y.append(species)

    return np.array(X), np.array(y)