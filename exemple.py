from ChouetteAPP import ModelHandler

def main():
    modelHandler = ModelHandler()
    modelHandler.train()
    modelHandler.loadWeights("2020-01-01", "2021-01-01")

    prediction = modelHandler.predict("data/2020-01-01_2021-01-01/vine/vine_0000.png")
    print(prediction)

    modelHandler.loadModel("ResNet50", "2020-01-01", "2021-01-01")
    prediction = modelHandler.predict("data/2020-01-01_2021-01-01/vine/vine_0000.png")
    print(prediction)

if __name__ == "__main__":
    main()