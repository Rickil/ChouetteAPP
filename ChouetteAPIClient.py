import httpx
import os
import shutil
from tqdm import tqdm

#This class handles the communication with the Chouette API
class ChouetteAPIClient:

    def __init__(self, login_url, dataset_url, username, password):
        self.login_url = login_url
        self.dataset_url = dataset_url
        self.username = username
        self.password = password
        self.token = self.login(username, password)
    
    #Login the user to get his token
    def login(self, username, password):
        headers = {"Content-Type": "application/json"}
        response = httpx.post(url=self.login_url, json={"username": username, "password": password}, headers=headers)
        return response.json()['token']
    
    #Download and store an image according to its tag
    def getImage(self, url, save_path):
        response = httpx.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)
    
    #Get all urls associated to a given tag in the dataset
    def getAllUrlByTag(self, tag, start_date, end_date):
        parameters = f"?tag={tag}&start_date={start_date}&end_date={end_date}"
        url = self.dataset_url + parameters
        headers = {"Authorization": f"Token {self.token}"}
        response = httpx.get(url, headers=headers)
        urls = [item["media"] for item in response.json()]
        return urls
    
    #Download the dataset and stores it in a folder named "data"
    def getDataset(self, start_date, end_date):
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.makedirs("data")

        tags = ["vine", "ground", "grass"]
        for tag in tqdm(tags, desc="Processing tags"):
            os.makedirs(f"data/{tag}")
            urls = self.getAllUrlByTag(tag, start_date, end_date)
            for i, image_url in enumerate(tqdm(urls, desc=f"Downloading {tag}", leave=False)):
                save_path = f"data/{tag}/{i:04d}.png"
                self.getImage(image_url, save_path)