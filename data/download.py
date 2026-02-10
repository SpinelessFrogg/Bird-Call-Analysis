import requests
from config import NATIVE_BIRDS, BATCH_DIR
import os

class XenoCantoClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def check_downloaded(self):
        birds_to_process = []
        for bird in NATIVE_BIRDS:
            batch_file = os.path.join(BATCH_DIR, f"{bird}_batch.npy")
            if not os.path.exists(batch_file):
                birds_to_process.append(bird)
        if not birds_to_process:
            return None
        else:
            return birds_to_process
    
    def get_recordings(self, bird_name):
        # pulls bird calls from xeno canto API
        recording_list = []
    
        # api call
        url = f'https://xeno-canto.org/api/3/recordings?query=en:"={bird_name}"&per_page=500&key={self.api_key}'
        response = requests.get(url)
        data = response.json()

        for rec in data.get("recordings", []):
            file = rec.get("file", "")
            url = f"https:{file}" if file.startswith("//") else file
            if url:
                recording_list.append(url)
        print(f"{bird_name} urls fetched.")
        return recording_list
    
    def get_bird_call_list(self, bird_list):
        bird_recordings = {}
        # API pull for bird calls
        for bird in bird_list:
            bird_recordings[bird] = self.get_recordings(bird)
        # returns list of lists
        return bird_recordings