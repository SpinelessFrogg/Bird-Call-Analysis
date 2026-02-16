import numpy as np
from concurrent.futures import ProcessPoolExecutor
from preprocessing.audio import load_mp3_url, decode_audiosegment
from preprocessing.features import waveform_to_melspec
import os
from config import MODEL_DIR

def url_to_spectrogram(url):
    audio = load_mp3_url(url)
    if audio is None:
        return None
    samples, sr = decode_audiosegment(audio)
    spec = waveform_to_melspec(samples, sr)
    if spec is None or spec.ndim != 2 or np.isnan(spec).any():
        return None
    return spec

def get_spectrogram_list(file_list):
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(url_to_spectrogram, file_list))
    return [r for r in results if r is not None]

def normalize(batch, mean=None, std=None, save_stats=False):
    # Normalize each spectrogram
    if mean is None or std is None:
        mean = np.mean(batch)
        std = np.std(batch) + 1e-9

        if save_stats:
            np.save(f"{MODEL_DIR}norm_mean.npy", mean)
            np.save(f"{MODEL_DIR}norm_std.npy", std)
    return (batch - mean) / std

def fix_width(spec, target_width=216):
    w = spec.shape[1]
    if w > target_width:
        return spec[:, :target_width]
    elif w < target_width:
        return np.pad(
            spec,
            ((0, 0), (0, target_width - w)),
            mode="constant"
        )
    return spec

def prepare_batch(specs, save_stats=False):
    X = np.array([fix_width(spec) for spec in specs], dtype=np.float32)
    X = normalize(X, save_stats=save_stats)
    X = np.expand_dims(X, axis=-1)
    return X

def prepare_single(spec):
    spec = fix_width(spec)
    
    mean = np.load(f"{MODEL_DIR}norm_mean.npy")
    std = np.load(f"{MODEL_DIR}norm_std.npy")

    spec = normalize(spec, mean, std)
    spec = np.expand_dims(spec, axis=-1)  # channel
    spec = np.expand_dims(spec, axis=0)   # batch
    return spec

def add_noise(X, scale=0.01):
    return X + scale * np.random.randn(*X.shape)

def save_spectrogram_DB(bird_name, spectrograms, save_dir="data/batches"):
    if not spectrograms:  # nothing to save
        return
    # save the raw data
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms))
    print(f'{bird_name} batch file created.')
