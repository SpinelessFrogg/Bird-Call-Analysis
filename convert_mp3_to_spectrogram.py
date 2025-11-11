import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import 

def mp3_to_spectrogram(mp3_file):
    # load the mp3 data
    data, sample_rate = librosa.load(mp3_file)

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

    # save the raw data
    save_dir = 


def display_spectrogram(spectrogram_in_dB, sample_rate):
    # displays the spectrogram
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram_in_dB, sr=sample_rate, x_axis='time', y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Bird Call")
    plt.show