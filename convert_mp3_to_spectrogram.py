import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_spectrogram(mp3_file):
    data, sample_rate = librosa.load(mp3_file)
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    spectrogram_in_dB = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram_in_dB, sr=sample_rate, x_axis='time', y_axis="mel")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram of Bird Call")
    print(spectrogram_in_dB)
    # plt.show()