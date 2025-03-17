# Spaceship Titanic Kaggle Competition Model

## Author: Alejandro Jiménez-González

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Description
This repository contains my current model for the Spaceship Titanic Kaggle competition. It uses GridSearchCV to find the best parameters between `RandomForestClassifier`, `CatBoostClassifier`, and `XGBClassifier`. Additionally, an `Artificial Neural Network` (ANN) is trained and compared. The best model is then trained and used to predict the test set. The predictions are saved in a CSV file in the 'output' folder.

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/alejimgon/Spaceship_Titanic-Kaggle-Competition-Model.git
    cd Spaceship_Titanic-Kaggle-Competition-Model
    ```

2. **Set up the conda environment**:
    ```sh
    conda create --name space_titanic python=3.12.7
    conda activate space_titanic
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Data
The dataset used in this competition is provided by Kaggle and contains information about the passengers on the Spaceship Titanic.

### Download the Data
1. **Download the data from Kaggle**:
    - Go to the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data) competition page on Kaggle.
    - Download the `train.csv` and `test.csv` files.

2. **Place the data files in the `data` directory**:
    - Create a `data` directory in the project root if it doesn't exist:
      ```sh
      mkdir data
      ```
    - Move the downloaded `train.csv` and `test.csv` files to the `data` directory.

## Model Training
The script performs the following steps:
1. **Data Preprocessing**: Handles missing values and encodes categorical variables.
2. **Feature Scaling**: Scales the features using `StandardScaler`.
3. **Grid Search**: Uses `GridSearchCV` to find the best parameters for `RandomForestClassifier`, `CatBoostClassifier`, and `XGBClassifier`.
4. **Artificial Neural Network**: Trains an Artificial Neural Network (ANN) model.
5. **Model Selection**: Compares the best scores and selects the best model.
6. **Model Training**: Trains the selected model with the best parameters.
7. **Prediction**: Uses the trained model to predict the test set.

## Usage
To run the script, use the following command:
```sh
python kaggle_space_titanic.py
```

## Results
The script outputs the best accuracy and parameters for each model. It also saves the predictions in a CSV file in the 'output' folder.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.