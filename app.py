from ChouetteAPP import ModelHandler, Trainer
import tensorflow as tf

def main():
    modelHandler = ModelHandler()
    trainer = Trainer()

    trainer.train(modelHandler)

if __name__ == "__main__":
    main()