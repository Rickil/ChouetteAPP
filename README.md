# ChouetteAPP

ChouetteAPP is a Python-based application designed for training and testing machine learning models on custom datasets. It provides easy-to-use configuration files and integrates with Sphinx for documentation generation. The application is structured as a module, with the core functionality encapsulated in a single class called `ModelHandler`, making it easy to reuse across different projects.

## Requirements

- Python 3.12.0
- A Database (PostgreSQL is supported; any SQL database with similar capabilities will also work)

### Installation

To get started with ChouetteAPP, you need to install the required dependencies.

1. Clone the repository or download the project files.
2. Install the necessary Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

3. After installing the dependencies, you can run the `example.py` file to test different features of the application:

   ```bash
   python example.py
   ```

## Code Structure

ChouetteAPP is structured as a reusable Python module, where the core functionality is encapsulated within the `ModelHandler` class. The module is located in the `ChouetteApp` directory, and it is designed for easy integration into other projects. 

### Main Class: `ModelHandler`

The `ModelHandler` class is the heart of the application. By creating an instance of this class, you can easily perform tasks and reuse it.

### Model Weights and Database Integration

Model weights are stored on disk, and the database holds the paths to these weights. This allows you to manage and retrieve model weights efficiently. The database can be a PostgreSQL database, as specified in the `.env.example` file, or any other SQL database that supports similar operations.

## Configuration Files

There are two main configuration files used by ChouetteAPP:

### 1. `train_config.json`

This file contains parameters for the dataset, training configuration, and model compilation settings.

```json
{
    "dataset_params": {
        "path": "data",  // Path to store the dataset
        "start_date": "2020-01-01",
        "end_date": "2021-01-01"
    },
    "compile_params": {
        "optimizer": "adam",
        "loss": "CategoricalCrossentropy",
        "metrics": ["accuracy"]
    },
    "training_params": {
        "image_size": [256, 256],
        "batch_size": 8,
        "validation_split": 0.2,
        "epochs": 5
    }
}
```

- `dataset_params`: Specifies the dataset storage path and the date range for the dataset.
- `compile_params`: Defines the optimizer, loss function, and evaluation metrics for the model.
- `training_params`: Configures the training settings, such as image size, batch size, validation split, and epochs.

### 2. `model_config.json`

This file defines the model architecture and its configurations.

```json
{
    "name": "ResNet50",
    "weights": null,
    "input_shape": [256, 256, 3],
    "classes": 3,
    "class_names": ["grass", "ground", "vine"]
}
```

- `name`: Specifies the model architecture (e.g., `ResNet50`).
- `weights`: Defines the pretrained weights to use. Set to `null` if no pretrained weights are desired, or use a date range format `"YYYY-MM-DD:YYYY-MM-DD"` to select weights from a specific dataset which are already in the database.
- `input_shape`: Specifies the input shape for the model.
- `classes`: The number of output classes for the model.
- `class_names`: A list of class names for classification.

## Documentation

ChouetteAPP uses Sphinx to generate documentation. To build the documentation in HTML format, follow these steps:

1. Go to the `docs` directory and create the `build` directory:

   ```bash
   cd docs
   mkdir build
   ```

2. Go to the `source` directory:

   ```bash
   cd source
   ```

3. Run Sphinx to build the documentation:

   ```bash
   sphinx-build -b html . ../build/html
   ```

4. After building the documentation, you can view it in HTML format by opening the following file in a browser:

   ```bash
   docs/build/html/index.html
   ```