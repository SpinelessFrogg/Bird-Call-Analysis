# hoo boy this is getting complicated
import librosa
from librosa import resample
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from io import BytesIO
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

def _load_mp3_url(url, target_sample_rate=22050):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    # complicated audio processing stuff
    audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)  # mono
    samples /= np.iinfo(audio.array_type).max  # normalize to [-1,1]

    # Resample to consistent sample rate for librosa
    if audio.frame_rate != target_sample_rate:
        samples = resample(samples, orig_sr=audio.frame_rate, target_sr=target_sample_rate)
        return samples, target_sample_rate
    return samples, audio.frame_rate

def trim_by_frequency(data, sample_rate, threshold_dB=20, max_length=5):
    # trims the data 
    trimmed, _ = librosa.effects.trim(data, top_db=threshold_dB)
    target_length = sample_rate * max_length
    return librosa.util.fix_length(trimmed, size=target_length)

def mp3_to_spectrogram(file):
    data, sample_rate = _load_mp3_url(file)

    # trims data
    data_trimmed = trim_by_frequency(data, sample_rate)

    # create the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=data_trimmed, sr=sample_rate)
    spectrogram_in_dB = librosa.power_to_db(spectrogram, ref=np.max)
    
    # # display
    # display_spectrogram(spectrogram_in_dB, sample_rate)

    # returns a spectrogram
    return spectrogram_in_dB

def get_spectrogram_list(file_list):
    # uses parallel processing to create the spectrograms
    with ThreadPoolExecutor(max_workers=8) as executor:
        spectrograms = list(executor.map(mp3_to_spectrogram, file_list))
    return spectrograms

def save_spectrogram_DB(bird_name, spectrograms, save_dir="batch_data"):
    # save the raw data
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms))

def display_spectrogram(spectrogram_in_dB, sample_rate):
    # displays the spectrogram
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram_in_dB, sr=sample_rate, x_axis='time', y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Bird Call")
    plt.show