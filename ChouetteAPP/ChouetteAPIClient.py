import httpx
import os
from tqdm import tqdm

#This class handles the communication with the Chouette API
class ChouetteAPIClient:

    def __init__(self):
        self.login_url = os.getenv("API_LOGIN_URL")
        self.dataset_url = os.getenv("API_DATASET_URL")
        self.username = os.getenv("API_USERNAME")
        self.password = os.getenv("API_PASSWORD")
        self.token = self.login(self.username, self.password)
    
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
    
    #Download the dataset and stores it in the given path
    def getDataset(self, path, start_date, end_date):
        dataset_path = os.path.join(path, f"{start_date}_{end_date}")
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

            tags = ["vine", "ground", "grass"]
            for tag in tqdm(tags, desc="Processing tags"):
                tag_path = os.path.join(dataset_path, tag)
                os.makedirs(tag_path)
                urls = self.getAllUrlByTag(tag, start_date, end_date)
                for i, image_url in enumerate(tqdm(urls, desc=f"Downloading {tag}", leave=False)):
                    image_path = os.path.join(tag_path, f"{tag}_{i:04d}.png")
                    self.getImage(image_url, image_path)
        
        return dataset_path