# from pydub import AudioSegment
import get_bird_calls
import convert_mp3_to_spectrogram

if __name__ == "__main__":
#     birds to pull from database
    native_birds = ["Scissor-tailed Flycatcher",
                    "Red-bellied Woodpecker",
                    "White-eyed Vireo"]
    # for testing
    # native_birds = ["Scissor-tailed Flycatcher"]
    
    all_bird_sounds = get_bird_calls.get_bird_call_list(native_birds)

    for bird in all_bird_sounds:
        list_of_sounds = all_bird_sounds[bird]
        species_spectrograms = convert_mp3_to_spectrogram.get_spectrogram_list(list_of_sounds)

        # for testing
        # convert_mp3_to_spectrogram.display_spectrogram_batch(species_spectrograms)

        convert_mp3_to_spectrogram.save_spectrogram_DB(bird, species_spectrograms)