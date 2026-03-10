import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import xeno_api_key, NATIVE_BIRDS
from data import download

def main():
    client = download.XenoCantoClient(api_key=xeno_api_key)
    print(len(NATIVE_BIRDS))
    for bird in NATIVE_BIRDS:
        client.get_good_recs(bird, 200)

main()