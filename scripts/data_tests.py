import numpy as np
import hashlib
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR
from preprocessing.dataset_builder import DatasetBuilder
from data.load_data import load_spectrogram_batches
from scripts.download_batches import main
import matplotlib.pyplot as plt
import random

X, y = load_spectrogram_batches()

builder = DatasetBuilder(X, y)
X_train_raw, X_test_raw, y_train, y_test = builder.split(X, y)
builder.X = X_train_raw
builder.y = y_train
X_train, y_train = builder.prepare()

builder.X = X_test_raw
builder.y = y_test
X_test, y_test = builder.prepare()
np.save(f"{MODEL_DIR}X_test.npy", X_test)
np.save(f"{MODEL_DIR}y_test.npy", y_test)

# are there near dupes?
def hash_spec(spec):
    return hashlib.md5(spec.tobytes()).hexdigest()
    
train_hashes = set(hash_spec(x) for x in X_train.squeeze())
val_hashes   = set(hash_spec(x) for x in X_test.squeeze())

overlap = train_hashes.intersection(val_hashes)

print("Exact duplicates:", len(overlap))

# same recordings
train_url_list, val_url_list = main()
train_urls = set(train_url_list)
val_urls   = set(val_url_list)

print("URL overlap:", len(train_urls.intersection(val_urls)))

# similar spectrograms
plt.figure(figsize=(10,6))

for i in range(3):
    idx = random.randint(0, len(X_train)-1)
    plt.subplot(2,3,i+1)
    plt.imshow(X_train[idx].squeeze(), aspect='auto')
    plt.title("Train")

for i in range(3):
    idx = random.randint(0, len(X_test)-1)
    plt.subplot(2,3,i+4)
    plt.imshow(X_test[idx].squeeze(), aspect='auto')
    plt.title("Val")

plt.show()

# distribution check
print("Train distribution:")
print(np.unique(y_train.argmax(axis=1), return_counts=True))

print("Val distribution:")
print(np.unique(y_test.argmax(axis=1), return_counts=True))

