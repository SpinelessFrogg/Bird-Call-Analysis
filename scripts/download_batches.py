from config import api_key
from data import download
from preprocessing.pipeline import get_spectrogram_list, save_spectrogram_DB

def main():
    getter = download.XenoCantoClient(api_key=api_key)
    recordings = getter.get_bird_call_list(bird_list=getter.check_downloaded())

    for bird, urls in recordings.items():
        specs = get_spectrogram_list(urls)
        save_spectrogram_DB(bird, specs)

if __name__ == "__main__":
    main()