import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/NEWDB/DB2labels/imputed_clevelanddata_2labels.csv')

# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create DataFrames for the resampled training data and the original test data
resampled_train_df = pd.DataFrame(data=np.c_[X_train_resampled, y_train_resampled], columns=df.columns)
test_df = pd.DataFrame(data=np.c_[X_test, y_test], columns=df.columns)

# Calculate the number of records for each label before SMOTE
before_smote_counts = pd.Series(y_train).value_counts()
print("Class distribution before SMOTE:")
print(before_smote_counts)

# Calculate the number of records for each label after SMOTE
after_smote_counts = pd.Series(y_train_resampled).value_counts()
print("Class distribution after SMOTE:")
print(after_smote_counts)

# Concatenate resampled training data and original test data
concatenated_df = pd.concat([resampled_train_df, test_df], ignore_index=True)

# Save the concatenated dataset to a CSV file
concatenated_df.to_csv('SMOTE_imputed_clevelanddata_2labels.csv', index=False)
