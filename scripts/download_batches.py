import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ebird_api_key
from data import download
from preprocessing.pipeline import get_spectrogram_list, save_spectrogram_DB

def main():
    # getter = download.XenoCantoClient(api_key=xeno_api_key)
    client = download.EBirdClient(api_key=ebird_api_key)
    to_process = client.check_downloaded()
    if to_process:
        recordings = client.get_bird_call_list(bird_list=to_process)
        for bird, urls in recordings.items():
            specs = get_spectrogram_list(urls)
            save_spectrogram_DB(bird, specs)
    else:
        print("No species to process.")

if __name__ == "__main__":
    main()