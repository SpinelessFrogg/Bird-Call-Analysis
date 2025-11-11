import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor

# pulls bird calls from xeno canto API
def get_bird_calls(bird_name):
    recording_list = []
    # # skip if all downloaded
    # if os.path.exists(f"Calls/{bird_name}"):
    #     return f"{bird_name} recordings already pulled."
    
    # api call
    url = f'https://xeno-canto.org/api/3/recordings?query=en:"={bird_name}"&key=ecebbcf8ef68a7e2bad17baf8e336ab6cc4c8d1d'
    response = requests.get(url)
    data = response.json()

    # create folder
    # os.makedirs(f"Calls/{bird_name}", exist_ok=True)

    for rec in data["recordings"]:
        url = f"https:{rec['file']}" if rec["file"].startswith("//") else rec["file"]
        recording_list.append(url)
    print(recording_list)

get_bird_calls("Scissor-tailed Flycatcher")