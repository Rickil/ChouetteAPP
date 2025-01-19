import httpx
import os
from tqdm import tqdm

#This class handles the communication with the Chouette API
class ChouetteAPIClient:
    """
    Handles communication with the Chouette API.

    Attributes:
        login_url (str): The URL for logging into the API.
        dataset_url (str): The URL for accessing the dataset.
        username (str): API username loaded from environment variables.
        password (str): API password loaded from environment variables.
        token (str): Authentication token retrieved after logging in.
    """

    def __init__(self):
        """
        Initializes the ChouetteAPIClient instance and logs in to obtain an authentication token.
        """
        self.login_url = os.getenv("API_LOGIN_URL")
        self.dataset_url = os.getenv("API_DATASET_URL")
        self.username = os.getenv("API_USERNAME")
        self.password = os.getenv("API_PASSWORD")
        self.token = self.login(self.username, self.password)

    def login(self, username, password):
        """
        Logs in to the API and retrieves an authentication token.

        Args:
            username (str): The username for the API.
            password (str): The password for the API.

        Returns:
            str: The authentication token.
        """
        headers = {"Content-Type": "application/json"}
        response = httpx.post(url=self.login_url, json={"username": username, "password": password}, headers=headers)
        return response.json()['token']
    
    def getImage(self, url, save_path):
        """
        Downloads and saves an image from a URL.

        Args:
            url (str): The URL of the image.
            save_path (str): The path to save the downloaded image.
        """
        response = httpx.get(url)
        with open(save_path, "wb") as file:
            file.write(response.content)
    
    def getAllUrlByTag(self, tag, start_date, end_date):
        """
        Retrieves all image URLs associated with a given tag within a date range.

        Args:
            tag (str): The tag to filter images.
            start_date (str): Start date in "YYYY-MM-DD" format.
            end_date (str): End date in "YYYY-MM-DD" format.

        Returns:
            list: A list of URLs.
        """
        parameters = f"?tag={tag}&start_date={start_date}&end_date={end_date}"
        url = self.dataset_url + parameters
        headers = {"Authorization": f"Token {self.token}"}
        response = httpx.get(url, headers=headers)
        urls = [item["media"] for item in response.json()]
        return urls
    
    def getDataset(self, path, start_date, end_date):
        """
        Downloads the dataset and stores it in the given path.

        Args:
            path (str): The directory path where the dataset will be stored.
            start_date (str): Start date in "YYYY-MM-DD" format.
            end_date (str): End date in "YYYY-MM-DD" format.

        Returns:
            str: The path to the downloaded dataset.
        """
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