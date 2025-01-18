from ChouetteAPP import ModelHandler, Trainer

def main():
    modelHandler = ModelHandler("ResNet50")
    trainer = Trainer()

    trainer.train(modelHandler.model)

if __name__ == "__main__":
    main()