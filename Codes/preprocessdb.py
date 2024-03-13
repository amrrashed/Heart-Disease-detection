import pandas as pd
import numpy as np

#processed.hungariandata
#processed.switzerlanddata
#processed.vadata
# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/datasets/DB2/heart+disease/reprocessed.hungariandata.txt',delimiter=' ')
#df=pd.read_csv('D:/new researches/Heart disease/datasets/db6/statlog+heart/heart.dat', delimiter=' ')
print(df.info())

# Assuming 'df' is your DataFrame
dataset_summary = df.describe(include='all')
print("Dataset Summary:")
print(dataset_summary)

df['class'] = df['class'].apply(lambda x: 1 if x != 0 else x)
# Replace '?' with NaN
df.replace('?', pd.NA, inplace=True)

# # Replace space values with NaN
#df.replace(' ', pd.NA, inplace=True)

# Count the number of missing values (NaN) in each column
missing_data_count = df.isna().sum()

# Calculate the percentage of missing values in each column
total_rows = len(df)
missing_data_percentage = (missing_data_count / total_rows) * 100

# Print the percentage of missing values in each column
print("Percentage of missing data in each column:")
print(missing_data_percentage)

# Impute missing values with the most frequent value in each column
df_imputed = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# Reset index after imputing values
df_imputed = df_imputed.reset_index(drop=True)

# Drop rows containing NaN values
#df.dropna(inplace=True)
df_imputed.to_csv('reprocessed.hungariandata_2labels.csv', index=False)
