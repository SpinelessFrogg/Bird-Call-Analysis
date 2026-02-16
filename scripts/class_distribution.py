import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_spectrogram_batches
import numpy as np

X, y = load_spectrogram_batches()

unique, counts = np.unique(y, return_counts=True)

for u, c in zip(unique, counts):
    print(f"{u}: {c}")

print("\nTotal samples:", len(y))