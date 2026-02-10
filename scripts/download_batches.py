import os
from data import download
from preprocessing.pipeline import get_spectrogram_list, save_spectrogram_DB
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

#     birds to pull from database
NATIVE_BIRDS = ["American Wigeon",
                "American Yellow Warbler",
                "Baltimore Oriole",
                "Bell's Vireo",
                "Black-capped Vireo",
                "Blue-grey Gnatcatcher",
                "Blue-winged Teal",
                "Burrowing Owl",
                "Carolina Chickadee",
                "Carolina Wren",
                "Chuck-will's-widow",
                "Common Grackle",
                "Red-bellied Woodpecker",
                "White-eyed Vireo"]

BATCH_DIR = "data/batches/"


def main():
    getter = download.XenoCantoClient(api_key=api_key)
    recordings = getter.get_bird_call_list(bird_list=getter.check_downloaded())

    for bird, urls in recordings.items():
        specs = get_spectrogram_list(urls)
        save_spectrogram_DB(bird, specs)

if __name__ == "__main__":
    main()
    