#!/usr/bin/env python
# coding: utf-8

# This Notebook will be used to prepare the Bicycle Trip Data 
# for use in Machine Learning Models

# Importing requisite libraries
import pandas as pd
import os

# Set up data path 
data_path = 'data'

# Import Specialized bike data to dataframe
print('Importing Specialized Data')
sp_data = 'SpecializedTrips.csv'
sp_df = pd.read_csv(os.path.join(data_path, sp_data))
#sp_df

# Add Label for Specialized Data
sp_df['Type'] = "Specialized"
#sp_df

# Import Citibike data to dataframe
print('Importing Citibike Data')
cb_data = 'CitibikeTrips.csv'
cb_df = pd.read_csv(os.path.join(data_path, cb_data))
#cb_df

# Add Label for Citibike Data
cb_df['Type'] = 'Citibike'
#cb_df

# Combine the dataframes
print('Combining Data')
all_data = [sp_df, cb_df]
all_df = pd.concat(all_data)
#all_df

# Sort the combined dataframe randomly
all_df = all_df.sample(frac=1).reset_index(drop=True)
#all_df

# Export to CSV file for use in Machine Learning Model
all_data_file = 'all_data.csv'
all_df.to_csv(os.path.join(data_path, all_data_file), index=False)

print("Bicycle Trip Data Import Complete")
