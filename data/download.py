import requests

class XenoCantoClient:
    def __init__(self, api_key):
        api_key = """#APIKEY"""

    def get_recordings(self, bird_name):
        # pulls bird calls from xeno canto API
        recording_list = []
    
        # api call
        url = f'https://xeno-canto.org/api/3/recordings?query=en:"={bird_name}"&per_page=500&key={self.api_key}'
        response = requests.get(url)
        data = response.json()

        # creates a list of all audio file urls
        # for rec in data["recordings"]:
        #     url = f"https:{rec['file']}" if rec["file"].startswith("//") else rec["file"]
        #     recording_list.append(url)
        # return recording_list

        for rec in data.get("recordings", []):
            file = rec.get("file", "")
            url = f"https:{file}" if file.startswith("//") else file
            if url:
                recording_list.append(url)
        return recording_list
    
    def get_bird_call_list(self, bird_list):
        bird_recordings = {}
        # API pull for bird calls
        for bird in bird_list:
            bird_recordings[bird] = self.get_recordings(bird)
        # returns list of lists
        return bird_recordings