#!/usr/bin/env python
# coding: utf-8

# This Machine Learning Model will Attempt to Predict the Type of Bicycle Used from Trip Data

# Importing Libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from IPython.display import display

# Set up data path
data_path = 'data'

# Load base data to dataframe
print('Loading Data . . . ')
all_data_file = 'all_data.csv'
all_df = pd.read_csv(os.path.join(data_path, all_data_file))
all_df

# Change NaN values to 0 - only occurs on hours column so this is accurate
all_df['Hours'] = all_df['Hours'].fillna(0)
all_df

# Turning Index into Orig_Index Column to trace back to original records
all_df.reset_index(inplace=True)
all_df = all_df.rename(columns = {'index': 'Orig_Index'})
all_df

# Defining fields to use
columns = ['Miles', 'Duration', 'Speed','Type']
# target = ['Type']

# Create Dataframe with desired columns
bike_df = all_df.loc[:,columns].copy()
bike_df

# Splitting out Features and Target
y = bike_df["Type"]
X = bike_df.drop(columns="Type")

# Splitting out Training and Testing data
# from sklearn.model_selection import train_test_split
print('Training Logistic Regression Model . . . ')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

# Checking training data shape
print('Training Shape:', X_train.shape)

# Checking testing data shape
print('Testing Shape:', X_test.shape)

# Creating the Logistic Regression Model
# from sklearn.linear_model import LogisticRegression
print('Running Logistic Regression Model . . .')
classifier = LogisticRegression(solver='lbfgs', max_iter=200, random_state=1)

# Training the Model
classifier.fit(X_train, y_train)

# Create Predictions
y_pred = classifier.predict(X_test)

# Get accuracy score
# from sklearn.metrics import accuracy_score
accu_score = accuracy_score(y_test, y_pred)

# Convert Predictions to DataFrame
pred_df = pd.DataFrame(y_pred)
pred_df.reset_index()
pred_df = pred_df.rename(columns = {0:'Prediction'})
#pred_df

# Convert Tests to DataFrame
test_df = pd.DataFrame(y_test)
#test_df

# Making index into a column
test_df.reset_index(inplace=True)
test_df = test_df.rename(columns = {'index': 'Orig_Index'})
#test_df

# Combining the new test and predition dataframes horizontally for comparison
test_pred_df = pd.concat([test_df, pred_df], axis = 1)
#test_pred_df

# Identify records where prediction is wrong
pred_errs_df = test_pred_df.loc[test_pred_df['Type'] != test_pred_df['Prediction']]
#pred_errs_df

# Merging pred_errs_df with all_df to see details of errors
err_details_df = pd.merge(pred_errs_df, all_df, on=["Orig_Index", "Orig_Index"])
#err_details_df

# Cleaning up err_details_df to eliminate redundancy and improve readability
err_details_df.drop(columns='Type_y', inplace=True)
err_details_df.rename(columns = {'Type_x' : 'Type'}, inplace=True)
print('Misclassified Records:')
display(err_details_df.to_string())

# Print formatted Accuracy Score
print('Accruacy Score is: {:.2f}'.format(accu_score * 100) + '%')
