import tensorflow as tf
import os
from datetime import datetime
from .utils import load_config, strToDate
from .database import Model, engine
from sqlalchemy import select
from sqlalchemy.orm import Session

class ModelHandler:
    def __init__(self):
        self.model_config = load_config("config/model_config.json")
        self.model_name = self.model_config["name"]
        self.model = None
        self.initModel()

    #Save the model weights in the database
    def save(self, start_date, end_date):
        print(start_date, end_date)
        os.makedirs("weights", exist_ok=True)
        weights_path = f"weights/{self.model_name}_{start_date}_{end_date}.weights.h5"
        self.model.save_weights(weights_path)

        session = Session(engine)
        model = Model(
            model_name=self.model_name,
            weights_path=weights_path,
            dataset_start_date=strToDate(start_date),
            dataset_end_date=strToDate(end_date),
            )
        
        session.add(model)
        session.commit()
    
    #Load model weights trained on a given dataset from the database
    def load(self, start_date, end_date):
        session = Session(engine)
        stmt = select(Model).where(Model.model_name == self.model_name,
                                   Model.dataset_start_date==strToDate(start_date),
                                     Model.dataset_end_date==strToDate(end_date))
        row = session.execute(stmt).fetchone()

        if row is None:
            raise ValueError(f"No model found with name {self.model_name} and dates {start_date} to {end_date}")
        weights_path = row[0].weights_path
        self.model.load_weights(weights_path)

    
    #Initialize the right version of the model
    def initModel(self):
        # Mapping model names to their corresponding Keras applications
        model_map = {
            "ResNet50": tf.keras.applications.ResNet50,
            "ResNet101": tf.keras.applications.ResNet101,
            "ResNet152": tf.keras.applications.ResNet152,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "ResNet101V2": tf.keras.applications.ResNet101V2,
            "ResNet152V2": tf.keras.applications.ResNet152V2
        }

        if self.model_name in model_map:
            self.model = model_map[self.model_name](
                include_top=True,
                input_shape = self.model_config["input_shape"],
                weights=None,
                classes = self.model_config["classes"]
            )
        else:
            raise ValueError(f"model {self.model_name} is unavailable !")
        
        if self.model_config["weights"]:
            start_date, end_date = self.model_config["weights"].split(":")
            self.load(start_date, end_date)