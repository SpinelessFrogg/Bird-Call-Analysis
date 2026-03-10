import numpy as np
from concurrent.futures import ProcessPoolExecutor
from preprocessing.audio import load_mp3_url, decode_audiosegment
from preprocessing.features import waveform_to_melspec
import os
import tensorflow as tf
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
    spectrograms = []
    for url, spec in zip(file_list, results):
        if spec is not None:
            spectrograms.append({
                "url": url,
                "spec": spec
            })
    return spectrograms

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

def prepare_batch(X, save_stats=False):
    X = np.array([fix_width(spec) for spec in X], dtype=np.float32)
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

def save_spectrogram_DB(bird_name, spectrograms, save_dir="data/batches"):
    if not spectrograms:  # nothing to save
        return
    # save the raw data
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms, dtype=object))
    print(f'{bird_name} batch file created.')

def augment_spec(spec, label):
    # freq masking
    freq_mask_size = tf.random.uniform((), 0, 20, dtype=tf.int32)
    freq_start = tf.random.uniform((), 0, 128 - freq_mask_size, dtype=tf.int32)
    freq_mask = tf.concat([
        tf.ones([freq_start, tf.shape(spec)[1], 1]),
        tf.zeros([freq_mask_size, tf.shape(spec)[1], 1]),
        tf.ones([128 - freq_start - freq_mask_size, tf.shape(spec)[1], 1])
    ], axis=0)
    spec = spec * freq_mask

    # time masking
    time_mask_size = tf.random.uniform((), 0, 30, dtype=tf.int32)
    time_start = tf.random.uniform((), 0, 216 - time_mask_size, dtype=tf.int32)
    time_mask = tf.concat([
        tf.ones([tf.shape(spec)[0], time_start, 1]),
        tf.zeros([tf.shape(spec)[0], time_mask_size, 1]),
        tf.ones([tf.shape(spec)[0], 216 - time_start - time_mask_size, 1])
    ], axis=1)
    spec = spec * time_mask

    # Gaussian noise
    spec = spec + tf.random.normal(tf.shape(spec), stddev=0.02)

    return spec, label

def runtime_augment(X, y, augment=False, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        ds = ds.map(augment_spec, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)