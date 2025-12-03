import numpy as np
import os

# load
data_dir = "batch_data"
spects = []
labels = []

for file in sorted(os.listdir(data_dir)):
    if not file.endswith(".npy"):
        continue
    species = file.replace("_batch.npy", "")
    arr = np.load(os.path.join(data_dir, file), allow_pickle=True)
    print(f"{file}: {len(arr)} samples")
    for s in arr[:3]:    # show first 3 example shapes/min/max
        print("  sample shape:", np.array(s).shape, 
              "min/max:", float(np.min(s)), float(np.max(s)))

# load all into arrays (like your pipeline)
all_specs = []
all_labels = []
for file in sorted(os.listdir(data_dir)):
    if not file.endswith(".npy"):
        continue
    species = file.replace("_batch.npy", "")
    arr = np.load(os.path.join(data_dir, file), allow_pickle=True)
    for s in arr:
        all_specs.append(np.array(s))
        all_labels.append(species)

all_specs = np.array(all_specs, dtype=object)
print("Total samples:", len(all_specs))
# check for constant or NaN spectrograms
nan_count = 0
constant_count = 0
bad_type_count = 0

for i, s in enumerate(all_specs):
    s = np.array(s, dtype=np.float32)   # force conversion to numeric array

    if s.ndim != 2:   # not a valid spectrogram
        bad_type_count += 1
        continue

    if np.isnan(s).any():
        nan_count += 1

    if np.isclose(s.max(), s.min()):
        constant_count += 1

print("NaN spectrograms:", nan_count)
print("Constant spectrograms:", constant_count)
print("Invalid-type spectrograms:", bad_type_count)

# class distribution
from collections import Counter
print("Class counts:", Counter(all_labels))