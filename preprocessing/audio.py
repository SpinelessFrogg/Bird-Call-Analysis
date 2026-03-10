import requests
from pydub import AudioSegment, exceptions
from io import BytesIO
from librosa import effects, resample
import numpy as np
import random

def load_mp3_url(url):
    # pulls mp3 data or returns none if erroring
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BirdCallMLBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ReadTimeout:
        print(f"ReadTimeout occurred for {url}. Skipping this file.")
        return None

    if 'audio' not in response.headers.get('Content-Type', ''):
        return None
    try:
        audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    except exceptions.CouldntDecodeError:
        return None
    return audio

def decode_audiosegment(audio, target_sample_rate=22050):
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)  # mono
    samples /= np.iinfo(audio.array_type).max  # normalize to [-1,1]

    # Resample to consistent sample rate for librosa
    if audio.frame_rate != target_sample_rate:
        samples = resample(samples, orig_sr=audio.frame_rate, target_sr=target_sample_rate)
        return samples, target_sample_rate
    return samples, audio.frame_rate

def augment_waveform(samples, sr):
    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.randn(len(samples))
        samples = samples + 0.003 * noise
    # Background noise injection
    if random.random() < 0.5:
        background = np.random.randn(len(samples))
        snr_db = random.uniform(10, 30)
        signal_power = np.mean(samples ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        background = background * np.sqrt(noise_power / (np.mean(background ** 2) + 1e-9))
        samples = samples + background
    # Pitch shift
    if random.random() < 0.4:
        n_steps = random.uniform(-2, 2)
        samples = effects.pitch_shift(samples, sr=sr, n_steps=n_steps)
    # Time stretch
    if random.random() < 0.4:
        rate = random.uniform(0.85, 1.15)
        samples = effects.time_stretch(samples, rate=rate)
        target_len = int(5.0 * sr)
        if len(samples) > target_len:
            samples = samples[:target_len]
        elif len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))
    # Volume scaling
    if random.random() < 0.5:
        samples = samples * random.uniform(0.6, 1.4)

    return samples