import requests
import json
import os

def get_bird_calls(bird_name):
    if os.path.exists(f"Calls/{bird_name}"):
        return f"{bird_name} recordings already pulled."
    url = f'https://xeno-canto.org/api/3/recordings?query=en:"={bird_name}"&key=ecebbcf8ef68a7e2bad17baf8e336ab6cc4c8d1d'
    response = requests.get(url)
    data = response.json()

    os.makedirs(f"Calls/{bird_name}", exist_ok=True)

    for i in range(len(data['recordings'])):
        audio_data = requests.get(data['recordings'][i]['file'])
        filename = f"Calls/{bird_name}/{bird_name}_{i}.mp3"
        with open(filename, 'wb') as f:
            f.write(audio_data.content)

    return f"Downloaded {len(data['recordings'])} recordings for {bird_name}."