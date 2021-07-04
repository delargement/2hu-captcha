from pybooru import Danbooru
from dotenv import load_dotenv
import os
load_dotenv()
DL_DIRECTORY = os.environ.get("DL-DIRECTORY")

client = Danbooru('danbooru')

