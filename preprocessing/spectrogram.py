import librosa
import librosa.display
from librosa import resample
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from requests.exceptions import ReadTimeout
from io import BytesIO
from pydub import AudioSegment, exceptions
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math

def _load_mp3_url(url, target_sample_rate=22050):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BirdSoundBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except ReadTimeout:
        print(f"ReadTimeout occurred for {url}. Skipping this file.")
        return None, None
    # complicated audio processing stuff

    if 'audio' not in response.headers.get('Content-Type', ''):
        return None, None
    
    try:
        audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    except exceptions.CouldntDecodeError:
        return None, None
    
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)  # mono
    samples /= np.iinfo(audio.array_type).max  # normalize to [-1,1]

    # Resample to consistent sample rate for librosa
    if audio.frame_rate != target_sample_rate:
        samples = resample(samples, orig_sr=audio.frame_rate, target_sr=target_sample_rate)
        return samples, target_sample_rate
    return samples, audio.frame_rate

def extract_call_region(data, sample_rate, window_seconds=5.0, hop_seconds=0.5):
    # Find the 5s segment with max RMS energy.
    target_len = int(window_seconds * sample_rate)
    # stop if clip < 5s
    if len(data) <= target_len:
        return librosa.util.fix_length(data, size=target_len)
    hop = max(1, int(hop_seconds * sample_rate))
    max_energy, best = -1.0, 0
    for start in range(0, len(data) - target_len + 1, hop):
        segment = data[start:start + target_len]
        energy = float(np.max(librosa.feature.rms(y=segment)))
        if energy > max_energy:
            max_energy, best = energy, start
    return data[best:best + target_len]

def mp3_to_spectrogram(file):
    data, sample_rate = _load_mp3_url(file)
    if data is None or sample_rate is None:
        return None
    # trims data
    data_trimmed, _ = librosa.effects.trim(data, top_db=20)
    call_region = extract_call_region(data_trimmed, sample_rate)

    # create the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=call_region, sr=sample_rate)
    spectrogram_in_dB = librosa.power_to_db(spectrogram, ref=np.max)

    if spectrogram_in_dB is None:
        return None
    spectrogram_in_dB = np.array(spectrogram_in_dB, dtype=np.float32)
    # must be 2D
    if spectrogram_in_dB.ndim != 2:
        return None
    # must have positive range
    if np.isnan(spectrogram_in_dB).any():
        return None
    return spectrogram_in_dB
    
    # # display
    # display_spectrogram(spectrogram_in_dB, sample_rate)

def get_spectrogram_list(file_list):
    # uses parallel processing to create the spectrograms
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for spec in executor.map(mp3_to_spectrogram, file_list):
            if spec is not None:
                results.append(spec)
    return results

def save_spectrogram_DB(bird_name, spectrograms, save_dir="batch_data"):
    if not spectrograms:  # nothing to save
        return
    # save the raw data
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms))
    print(f'{bird_name} completed')

def display_spectrogram(spectrogram_in_dB, sample_rate):
    # displays the spectrogram
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram_in_dB, sr=sample_rate, x_axis='time', y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Bird Call")
    plt.show()

# for testing
def display_spectrogram_batch(spectrograms, sample_rate=22050, max_show=10, cols=5):
    """Display up to max_show spectrograms in a grid for quick visual testing."""
    if not spectrograms:
        return
    subset = spectrograms[:max_show]
    rows = math.ceil(len(subset) / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, spec in enumerate(subset):
        plt.subplot(rows, cols, i + 1)
        librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.title(f"Spec {i}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()