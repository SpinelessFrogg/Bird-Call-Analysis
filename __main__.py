# from pydub import AudioSegment
from get_bird_calls import get_bird_calls, os
from convert_mp3_to_spectrogram import mp3_to_spectrogram

if __name__ == "__main__":
    # birds to pull from database
    native_birds = ["Scissor-tailed Flycatcher",
                    "Red-bellied Woodpecker",
                    "White-eyed Vireo"]
    
    # API pull for bird calls
    # for bird in native_birds:
    #     print(get_bird_calls(bird))

    # test code for showing bird calls 
    bird_dir = f"Calls/{native_birds[2]}/"
    mp3_to_spectrogram(f"{bird_dir}{os.listdir(bird_dir)[3]}")
    
        