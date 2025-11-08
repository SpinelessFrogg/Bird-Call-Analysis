# from pydub import AudioSegment
from get_bird_calls import get_bird_calls, os
from convert_mp3_to_spectrogram import mp3_to_spectrogram

if __name__ == "__main__":
    native_birds = ["Scissor-tailed Flycatcher",
                    "Red-bellied Woodpecker",
                    "White-eyed Vireo"]
    
    # for bird in native_birds:
    #     print(get_bird_calls(bird))

    bird_dir = f"Calls/{native_birds[0]}/"
    mp3_to_spectrogram(f"{bird_dir}{os.listdir(bird_dir)[3]}")
        