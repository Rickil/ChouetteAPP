import json
import os
from dotenv import load_dotenv
from datetime import datetime

#load environment variables
load_dotenv()

#load configuration from json config files
def load_config(path):
    with open(path, "r") as file:
        return json.load(file)

#convert string date in the format "YYYY-MM-DD" to a date type
def strToDate(str):
    return datetime.strptime(str, "%Y-%m-%d").date()