# This script is my attent to solve the Spaceship Titanic competition on Kaggle
# The goal is to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# The dataset is composed of 2 files: train.csv and test.csv
# The script will perform the following steps:
# 1. Load the dataset
# 2. Handle missing values
# 3. Apply OneHotEncoding to the categorical columns
# 4. Apply LabelEncoding to the target variable
# 5. Split the dataset into Training and Test sets
# 6. Apply Feature Scaling
# 7. Perform Grid Search to find the best parameters for RandomForestClassifier, XGBClassifier and CatBoostClassifier
# 8. Trin an Artificial Neural Network (ANN) model
# 9. Evaluate the ANN model
# 10. Choose the best classifier
# 11. Train the best model with the best parameters
# 12. Make the Confusion Matrix
# 13. Predict the Test set results
# 14. Create the submission file

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Load the dataset
train_dataset = pd.read_csv(f'{data_folder}/train.csv')
test_dataset = pd.read_csv(f'{data_folder}/test.csv')

# Creating the X and y variables
X = train_dataset.iloc[:, 1:-2]
X_test = test_dataset.iloc[:, 1:-1]
y = train_dataset.iloc[:, -1]

# Extract the first letter of the 'Cabin' column to create a new 'Deck' column
X['Deck'] = X['Cabin'].str[0]
X_test['Deck'] = X_test['Cabin'].str[0]

# Extract the type of type of the 'Cabin' column to create a new 'RoomType' column
X['RoomType'] = X['Cabin'].str[-1]
X_test['RoomType'] = X_test['Cabin'].str[-1]

# Drop the 'Cabin' column as it is no longer needed
X = X.drop(columns=['Cabin'])
X_test = X_test.drop(columns=['Cabin'])

# Identifying the columns with missing values
columns_with_mean_imputation = ['Age']
columns_with_zero_imputation = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
columns_with_unknown_imputation = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'RoomType']

# Handling missing values with mean imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[columns_with_mean_imputation] = mean_imputer.fit_transform(X[columns_with_mean_imputation])
X_test[columns_with_mean_imputation] = mean_imputer.transform(X_test[columns_with_mean_imputation])

# Handling missing values with zero imputation
zero_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X[columns_with_zero_imputation] = zero_imputer.fit_transform(X[columns_with_zero_imputation])
X_test[columns_with_zero_imputation] = zero_imputer.transform(X_test[columns_with_zero_imputation])

# Convert boolean columns to strings before 'Unknown' imputation
for col in ['CryoSleep', 'VIP']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Handling missing values with 'Unknown' imputation
none_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')
X[columns_with_unknown_imputation] = none_imputer.fit_transform(X[columns_with_unknown_imputation])
X_test[columns_with_unknown_imputation] = none_imputer.transform(X_test[columns_with_unknown_imputation])

# Applying OneHotEncoder to the categorical columns
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'RoomType']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), categorical_columns)], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.transform(X_test))

# Applying Label Encoding to the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#import sys
#sys.exit("Script stop for testing purposes")

# Splitting the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Parameters for Grid Search
rf_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

catboost_parameters = {
    'iterations': [50, 100, 200],
    'depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7]
}

xgboost_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0, 0.5, 1]
}

# Set the number of folds for cross-validation
cv_folds = 10

# Grid Search for RandomForestClassifier
print("Starting Grid Search for RandomForestClassifier")
rf_classifier = RandomForestClassifier(random_state=0)
rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_parameters, scoring='accuracy', cv=cv_folds, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
rf_best_accuracy = rf_grid_search.best_score_
rf_best_parameters = rf_grid_search.best_params_
print("RandomForest Best Accuracy: {:.2f} %".format(rf_best_accuracy*100))
print("RandomForest Best Parameters:", rf_best_parameters)

# Grid Search for XGBClassifier
print("Starting Grid Search for XGBClassifier")
xgb_classifier = XGBClassifier(random_state=0, verbosity=0)
xgb_grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=xgboost_parameters, scoring='accuracy', cv=cv_folds, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
xgb_best_accuracy = xgb_grid_search.best_score_
xgb_best_parameters = xgb_grid_search.best_params_
print("XGBClassifier Best Accuracy: {:.2f} %".format(xgb_best_accuracy*100))
print("XGBClassifier Best Parameters:", xgb_best_parameters)

# Grid Search for CatBoostClassifier
print("Starting Grid Search for CatBoostClassifier")
catboost_classifier = CatBoostClassifier(random_state=0, verbose=0)
catboost_grid_search = GridSearchCV(estimator=catboost_classifier, param_grid=catboost_parameters, scoring='accuracy', cv=cv_folds, n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
catboost_best_accuracy = catboost_grid_search.best_score_
catboost_best_parameters = catboost_grid_search.best_params_
print("CatBoostClassifier Best Accuracy: {:.2f} %".format(catboost_best_accuracy*100))
print("CatBoostClassifier Best Parameters:", catboost_best_parameters)

# Train the ANN model
print("Training ANN model")
ann_classifier = tf.keras.models.Sequential()
ann_classifier.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
ann_classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann_classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann_classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_classifier.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the ANN model
_, ann_best_accuracy = ann_classifier.evaluate(X_val, y_val)
print("ANN Best Accuracy: {:.2f} %".format(ann_best_accuracy*100))

# Choosing the best classifier
if rf_best_accuracy > xgb_best_accuracy and rf_best_accuracy > catboost_best_accuracy and rf_best_accuracy > ann_best_accuracy:
    best_classifier = RandomForestClassifier(**rf_best_parameters, random_state=0)
    print("Using RandomForestClassifier")
elif xgb_best_accuracy > rf_best_accuracy and xgb_best_accuracy > catboost_best_accuracy and xgb_best_accuracy > ann_best_accuracy:
    best_classifier = XGBClassifier(**xgb_best_parameters, random_state=0, verbosity=0)
    print("Using XGBClassifier")
elif catboost_best_accuracy > rf_best_accuracy and catboost_best_accuracy > xgb_best_accuracy and catboost_best_accuracy > ann_best_accuracy:
    best_classifier = CatBoostClassifier(**catboost_best_parameters, random_state=0, verbose=0)
    print("Using CatBoostClassifier")
else:
    best_classifier = ann_classifier
    print("Using ANN")

# Train the best model with the best parameters (if not ANN)
if best_classifier != ann_classifier:
    best_classifier.fit(X_train, y_train)

# Making the Confusion Matrix with the best model
y_pred = best_classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
print(cm)

# Predicting the Test set results with the best model
y_pred_test = best_classifier.predict(X_test)

# Convert the predictions back to the original format
y_pred_test = label_encoder.inverse_transform(y_pred_test)

# Creating the submission file
submission = pd.DataFrame({'PassengerId': test_dataset['PassengerId'], 'Transported': y_pred_test})
submission.to_csv(f'{data_folder}/output/submission.csv', index=False)
print("Predictions saved to submission.csv")
