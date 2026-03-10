from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

xeno_api_key = os.getenv("X_API_KEY")
ebird_api_key = os.getenv("E_API_KEY")

#     birds to pull from database
NATIVE_BIRDS = [
                "American Yellow Warbler",
                "Baltimore Oriole",
                "Bell's Vireo",
                "Blue-grey Gnatcatcher",
                "Carolina Chickadee",
                "Carolina Wren",
                "Common Grackle",
                "Red-bellied Woodpecker",
                "White-eyed Vireo",
                "Northern Cardinal",
                "American Robin",
                "American Crow",
                "Blue Jay",
                "Northern Mockingbird",
                "Red-winged Blackbird",
                "Mourning Dove",
                "Downy Woodpecker",
                "Hairy Woodpecker",
                "Northern Flicker",
                "Great Crested Flycatcher",
                "Red-eyed Vireo",
                "Yellow-throated Vireo",
                "House Wren",
                "Bewick's Wren",
                "Grey Catbird",
                "Brown Thrasher",
                "Eastern Bluebird",
                "Wood Thrush",
                "Swainson's Thrush",
                "American Goldfinch",
                "House Finch",
                "Indigo Bunting",
                "Dickcissel",
                "Eastern Towhee",
                "Spotted Towhee",
                "Song Sparrow",
                "Field Sparrow",
                "Chipping Sparrow",
                "Lark Sparrow",
                "Common Yellowthroat",
                "Yellow-breasted Chat",
                "Ovenbird",
                "Black-and-white Warbler",
                "Pine Warbler",
                "Hooded Warbler",
                "Kentucky Warbler",
                "Eastern Meadowlark",
                "Western Meadowlark",
                "Barn Swallow",
                "Killdeer",
                ]

BATCH_DIR = "data/batches/"

MODEL_DIR = "models/"

TAXONOMY_FILE = Path("data/taxonomy.json")
CODE_FILE = Path("data/coded_taxonomy.json")
MACAULAY_URL = "https://search.macaulaylibrary.org/api/v1/search"