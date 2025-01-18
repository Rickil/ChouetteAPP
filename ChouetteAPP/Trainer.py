import tensorflow as tf
from .utils import load_config
from .ChouetteAPIClient import ChouetteAPIClient

#This class is a model agnostic trainer
class Trainer:
    def __init__(self):
        self.APIClient = ChouetteAPIClient()

        train_config = load_config("config/train_config.json")
        self.dataset_params = train_config["dataset_params"]
        self.compile_params = train_config["compile_params"]
        self.training_params = train_config["training_params"]

        self.train_dataset = None
        self.validation_dataset = None
        self.loadDataset()

    #Doawnload the dataset and load it
    def loadDataset(self):
        self.APIClient.getDataset(self.dataset_params["start_date"], 
                                  self.dataset_params["end_date"])
        
        self.train_dataset, self.validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.dataset_params["path"],
            labels="inferred",
            label_mode = "categorical",
            color_mode="rgb",
            batch_size=self.training_params['batch_size'],
            image_size=self.training_params['image_size'],
            shuffle=True,
            seed=42,
            validation_split=self.training_params['validation_split'],
            subset="both",
            verbose=True
            )
    
    #Start the training of the model
    def train(self, model):
        model.compile(**self.compile_params)
        model.fit(
            self.train_dataset,
            validation_data= self.validation_dataset,
            epochs=self.training_params["epochs"]
        )