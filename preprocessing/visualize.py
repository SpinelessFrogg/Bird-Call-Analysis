import matplotlib.pyplot as plt
import librosa
import math

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