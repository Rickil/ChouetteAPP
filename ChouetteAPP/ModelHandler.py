import tensorflow as tf
from .utils import load_config

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_config = load_config("config/model_config.json")
        self.model = self.initModel()
    
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
            return model_map[self.model_name](**self.model_config)
        else:
            raise ValueError(f"model {self.model_name} is unavailable !")