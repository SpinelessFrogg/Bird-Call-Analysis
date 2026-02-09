import requests
from pydub import AudioSegment, exceptions
from io import BytesIO
from librosa import resample
import numpy as np

def _load_mp3_url(url):
    # pulls mp3 data or returns none if erroring
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BirdCallMLBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ReadTimeout:
        print(f"ReadTimeout occurred for {url}. Skipping this file.")
        return None, None

    if 'audio' not in response.headers.get('Content-Type', ''):
        return None, None
    try:
        audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
    except exceptions.CouldntDecodeError:
        return None, None
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