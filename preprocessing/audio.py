import requests
from pydub import AudioSegment, exceptions
from io import BytesIO
from librosa import resample
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

def augment_waveform(samples, sr, rarity=0, base_prob=0.4):
    if random.random() < 0.5:
        noise = np.random.randn(len(samples))
        samples = samples + 0.003 * noise

    if random.random() < 0.5:
        shift = int(0.1 * sr)
        samples = np.roll(samples, shift)

    if random.random() < 0.5:
        samples = samples * random.uniform(0.8, 1.2)

    return samples