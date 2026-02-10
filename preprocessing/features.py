import librosa
import numpy as np

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

def waveform_to_melspec(data, sample_rate):
    data_trimmed, _ = librosa.effects.trim(data, top_db=20)
    call_region = extract_call_region(data_trimmed, sample_rate)

    # create the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=call_region, sr=sample_rate, n_mels=128, n_fft=2048, hop_length=512)
    spectrogram_in_dB = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_in_dB.astype(np.float32)