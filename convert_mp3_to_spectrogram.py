import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from io import BytesIO
from pydub import AudioSegment

def _load_mp3_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)  # mono
    samples /= np.iinfo(audio.array_type).max  # normalize to [-1,1]
    return samples, audio.frame_rate

def mp3_to_spectrogram(file):
    data, sample_rate = _load_mp3_url(file)

    # finds the highest volume (bird chirps) and trims based on that at 5 seconds for regularity
    data_trimmed, index = librosa.effects.trim(data, top_db=20)
    target_length = sample_rate * 5  # 5 seconds
    if len(data_trimmed) > target_length:
        data_trimmed = data_trimmed[:target_length]
    else:
        data_trimmed = librosa.util.fix_length(data_trimmed, target_length)

    # create the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=data_trimmed, sr=sample_rate)
    spectrogram_in_dB = librosa.power_to_db(spectrogram, ref=np.max)
    
    # # display
    # display_spectrogram(spectrogram_in_dB, sample_rate)

    # returns a spectrogram
    return spectrogram_in_dB

def get_spectrogram_list(file_list):
    spectrogram_list = []
    for file in file_list:
        spectrogram_list.append(mp3_to_spectrogram(file))
    # returns a list of spectrograms for a type of bird
    return spectrogram_list

def save_spectrogram_DB(bird_name, spectrograms, save_dir="batch_data"):
    # save the raw data (WIP)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{bird_name}_batch.npy", np.array(spectrograms))

def display_spectrogram(spectrogram_in_dB, sample_rate):
    # displays the spectrogram
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram_in_dB, sr=sample_rate, x_axis='time', y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Bird Call")
    plt.show