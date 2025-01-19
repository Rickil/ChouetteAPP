import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
from .utils import load_config, strToDate
from .database import Model, engine
from .Trainer import Trainer
from sqlalchemy import select
from sqlalchemy.orm import Session

class ModelHandler:
    def __init__(self):
        self.model_config = load_config("config/model_config.json")
        self.model_name = self.model_config["name"]
        self.class_names = self.model_config["class_names"]
        self.model = None
        self.initModel()

        self.trainer = Trainer()

    #Save the model weights in the database
    def save(self):
        start_date = self.trainer.dataset_params["start_date"]
        end_date = self.trainer.dataset_params["end_date"]
        os.makedirs("weights", exist_ok=True)
        weights_path = os.path.join("weights", f"{self.model_name}_{start_date}_{end_date}.weights.h5")
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
    def loadWeights(self, start_date, end_date):
        session = Session(engine)
        stmt = select(Model).where(Model.model_name == self.model_name,
                                   Model.dataset_start_date==strToDate(start_date),
                                     Model.dataset_end_date==strToDate(end_date))
        row = session.execute(stmt).fetchone()

        if row is None:
            raise ValueError(f"No model found with name {self.model_name} and dates {start_date} to {end_date}")
        weights_path = row[0].weights_path
        self.model.load_weights(weights_path)
    
    #Load a new model from the database with pretrained weights
    def loadModel(self, model_name, start_date, end_date):
        self.model_name = model_name
        self.initModel()
        self.loadWeights(start_date, end_date)

    
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
    
    def train(self, save=True):
        self.trainer.train(self.model)
        if save:
            self.save()
    
    def predict(self, image_path):
        image = load_img(image_path, target_size=self.model_config["input_shape"][:2])
        image = img_to_array(image)
        image = image / 255.0
        image = tf.convert_to_tensor(image)
        image = tf.expand_dims(image, axis=0)
        result = self.model(image)
        prediction = tf.argmax(result, axis=-1).numpy()[0]
        class_prediction = self.class_names[prediction]

        return class_prediction