import requests
from concurrent.futures import ThreadPoolExecutor

# pulls bird calls from xeno canto API
def get_bird_calls(bird_name):
    recording_list = []
    
    # api call
    url = f'https://xeno-canto.org/api/3/recordings?query=en:"={bird_name}"&key=ecebbcf8ef68a7e2bad17baf8e336ab6cc4c8d1d'
    response = requests.get(url)
    data = response.json()

    # creates a list of all audio file urls
    for rec in data["recordings"]:
        url = f"https:{rec['file']}" if rec["file"].startswith("//") else rec["file"]
        recording_list.append(url)
    return recording_list

def get_bird_call_list(bird_list):
    bird_recordings = {}
    # API pull for bird calls
    for bird in bird_list:
        bird_recordings[bird] = get_bird_calls(bird)
    # returns list of lists
    return bird_recordings