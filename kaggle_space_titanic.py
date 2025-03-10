# This script is my attent to solve the Spaceship Titanic competition on Kaggle
# The goal is to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# The dataset is composed of 2 files: train.csv and test.csv

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Load the dataset
train_dataset = pd.read_csv(f'{data_folder}/train.csv')
test_dataset = pd.read_csv(f'{data_folder}/test.csv')

# Analyzing the dataset
#print(train_dataset.head())
#print(train_dataset.describe())
#print(train_dataset.info())
#print(train_dataset.count())

# Creating the X and y variables
X = train_dataset.iloc[:, 1:-2]
y = train_dataset.iloc[:, -1]

# Identifying the columns with missing values
columns_with_mean_imputation = ['Age']
columns_with_zero_imputation = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
columns_with_unknwon_imputation =['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']

# Handling missing values with mean imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[columns_with_mean_imputation] = mean_imputer.fit_transform(X[columns_with_mean_imputation])
test_dataset[columns_with_mean_imputation] = mean_imputer.transform(test_dataset[columns_with_mean_imputation])

# Handling missing values with zero imputation
zero_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X[columns_with_zero_imputation] = zero_imputer.fit_transform(X[columns_with_zero_imputation])
test_dataset[columns_with_zero_imputation] = zero_imputer.transform(test_dataset[columns_with_zero_imputation])

# Handling missing values with 'Unkown' imputation
none_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknwon')
X[columns_with_unknwon_imputation] = none_imputer.fit_transform(X[columns_with_unknwon_imputation])
test_dataset[columns_with_unknwon_imputation] = none_imputer.transform(test_dataset[columns_with_unknwon_imputation])


