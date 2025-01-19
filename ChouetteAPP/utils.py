import json
import os
from dotenv import load_dotenv
from datetime import datetime

#load environment variables
load_dotenv()

def load_config(path):
    """
    Load configuration from a JSON file.

    Args:
        path (str): The path to the JSON configuration file.

    Returns:
        dict: The parsed JSON configuration as a dictionary.
    """
    with open(path, "r") as file:
        return json.load(file)

def strToDate(str):
    """
    Convert a string in the format "YYYY-MM-DD" to a `datetime.date` object.

    Args:
        str (str): The date string in "YYYY-MM-DD" format.

    Returns:
        datetime.date: The converted date object.
    """
    return datetime.strptime(str, "%Y-%m-%d").date()