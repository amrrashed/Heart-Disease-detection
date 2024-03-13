import pandas as pd
import numpy as np


# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/datasets/DB2/heart+disease/processed.switzerlanddata.txt')
# Replace '?' with NaN
df = df.replace('?', pd.NA)

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

# Save the imputed DataFrame to a CSV file
df_imputed.to_csv("imputed_switzerlanddata_data.csv", index=False)



