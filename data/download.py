import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NATIVE_BIRDS, BATCH_DIR, MACAULAY_URL, CODE_FILE
import json


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
    
class EBirdClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def cache_taxonomy(self):
        headers = {"X-eBirdApiToken": self.api_key}
        r = requests.get(
            "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json",
            headers=headers
        )
        r.raise_for_status()
        tax = r.json()
        name_to_code = {t["comName"].lower(): t["speciesCode"] for t in tax}
        with open(CODE_FILE, "w") as f:
            json.dump(name_to_code, f)
        return name_to_code

    def load_coded_taxonomy(self):
        if CODE_FILE.exists():
            with open(CODE_FILE) as f:
                return json.load(f)

        return self.cache_taxonomy()

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
        
    def get_urls(self, code): # This function is heavily modified code from https://github.com/Md-Shaid-Hasan-Niloy/Macaulay_downloader
        # Tried to fork his repo and use it but it was giving me too many issues. Shoutout Md Shaid Hasan Niloy <3
        print(f"Searching for urls...")

        url = MACAULAY_URL
        page = 1
        url_list = []
        try:
            while page < 34:
                params = {
                            "taxonCode": code,
                            "mediaType": "audio",
                            "sort": "rating_rank_desc",
                            "pageSize": 1000,
                            "page": page
                }
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                json_data = response.json()
                results = json_data.get('results', {}).get('content', [])
                if not results:
                    break
                
                for item in results:
                    audio_url = item.get("audioUrl") or item.get("mediaUrl")
                    url_list.append(audio_url)
                page += 1
                
            print(f"Found {len(url_list)} total recordings on Macaulay")
            return url_list
            
        except Exception as e:
            print(e)
    
    def get_bird_call_list(self, bird_list):
        bird_recordings = {}
        # API pull for bird calls
        taxonomy = self.load_coded_taxonomy()

        for bird in bird_list:
            code = taxonomy.get(bird.lower())
            bird_recordings[bird] = self.get_urls(code=code)
            if not code:
                print(f"No code for {bird}")
                continue

        return bird_recordings