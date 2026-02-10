import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import api_key
from data import download
from preprocessing.pipeline import get_spectrogram_list, save_spectrogram_DB

def main():
    getter = download.XenoCantoClient(api_key=api_key)
    to_process = getter.check_downloaded()
    if to_process:
        recordings = getter.get_bird_call_list(bird_list=to_process)
        for bird, urls in recordings.items():
            specs = get_spectrogram_list(urls)
            save_spectrogram_DB(bird, specs)
    else:
        print("No species to process.")

if __name__ == "__main__":
    main()