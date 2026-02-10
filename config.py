from dotenv import load_dotenv
import os

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