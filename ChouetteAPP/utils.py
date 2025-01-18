import json
import os
from dotenv import load_dotenv

load_dotenv()

def load_config(path):
    with open(path, "r") as file:
        return json.load(file)