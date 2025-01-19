import tensorflow as tf
import os
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
        self.test_dataset = None
        self.loadDataset()

    #Download the dataset and load it
    def loadDataset(self, path=None):
        #If no path variable is provided, it will download the dataset from tha API, 
        #otherwise it will load the dataset in the given path
        if not path:
            path = self.APIClient.getDataset(self.dataset_params["path"],
                                    self.dataset_params["start_date"], 
                                    self.dataset_params["end_date"])
        
        self.train_dataset, self.validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
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

        #normalize the dataset
        normalization = tf.keras.layers.Rescaling(1./255)
        self.train_dataset = self.train_dataset.map(lambda x, y: (normalization(x), y))
        self.validation_dataset = self.validation_dataset.map(lambda x, y: (normalization(x), y))

    #Start the training of the model
    def train(self, model):

        # Map loss function strings to actual Keras loss functions
        loss_function_map = {
            "CategoricalCrossentropy": tf.losses.CategoricalCrossentropy,
            # Add more as needed
        }

        model.compile(
            optimizer=self.compile_params["optimizer"],
            metrics=self.compile_params["metrics"],
            loss=loss_function_map[self.compile_params["loss"]]()
        )

        model.fit(
            self.train_dataset,
            validation_data= self.validation_dataset,
            epochs=self.training_params["epochs"]
        )     