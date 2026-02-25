from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

xeno_api_key = os.getenv("X_API_KEY")
ebird_api_key = os.getenv("E_API_KEY")

#     birds to pull from database
NATIVE_BIRDS = ["American Wigeon",
                "Northern Yellow Warbler",
                "Baltimore Oriole",
                "Bell's Vireo",
                "Black-capped Vireo",
                "Blue-gray Gnatcatcher",
                "Blue-winged Teal",
                "Burrowing Owl",
                "Carolina Chickadee",
                "Carolina Wren",
                "Chuck-will's-widow",
                "Common Grackle",
                "Red-bellied Woodpecker",
                "White-eyed Vireo"]

BATCH_DIR = "data/batches/"

MODEL_DIR = "models/"

TAXONOMY_FILE = Path("data/taxonomy.json")
CODE_FILE = Path("data/coded_taxonomy.json")
MACAULAY_URL = "https://search.macaulaylibrary.org/api/v1/search"