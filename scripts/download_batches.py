import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import xeno_api_key, ebird_api_key
from data import download
from preprocessing.pipeline import get_spectrogram_list, save_spectrogram_DB
from sklearn.model_selection import train_test_split

def main():
    client = download.XenoCantoClient(api_key=xeno_api_key)
    # client = download.EBirdClient(api_key=ebird_api_key)
    to_process = client.check_downloaded()
    if to_process:
        recordings = client.get_bird_call_list(bird_list=to_process)
        for bird, urls in recordings.items():
            train_urls, test_urls = train_test_split(
                urls,
                test_size=0.2,
                random_state=42
            )
            train_specs = get_spectrogram_list(train_urls)
            test_specs = get_spectrogram_list(test_urls)
            save_spectrogram_DB(f"{bird}_train", train_specs)
            save_spectrogram_DB(f"{bird}_test", test_specs)
    else:
        print("No species to process.")

if __name__ == "__main__":
    main()