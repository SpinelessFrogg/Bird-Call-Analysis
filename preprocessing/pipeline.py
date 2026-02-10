import numpy as np
from concurrent.futures import ProcessPoolExecutor
from preprocessing.audio import load_mp3_url, decode_audiosegment
from preprocessing.features import waveform_to_melspec
import os


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

def save_spectrogram_DB(bird_name, spectrograms, save_dir="data/batches"):
    if not spectrograms:  # nothing to save
        return
    # save the raw data
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms))
    print(f'{bird_name} completed')
