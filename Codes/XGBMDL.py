import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/NEWDB/DB2labels/original/statlog.csv')


# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']
y=y-1

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# load the model from file
encoder = load_model('encoder4.h5')

# encode the train data
X_train_encode = encoder.predict(X_train)

# concatenate original and encoded data for training
X_train_merged = np.concatenate((X_train, X_train_encode), axis=1)

# encode the test data
X_test_encode = encoder.predict(X_test)

# concatenate original and encoded data for testing
X_test_merged = np.concatenate((X_test, X_test_encode), axis=1)

# Combine encoded train and test data with target labels
# Merge the data
train_data = np.concatenate((X_train, X_train_encode, np.expand_dims(y_train, axis=1)), axis=1)
test_data = np.concatenate((X_test, X_test_encode, np.expand_dims(y_test, axis=1)), axis=1)

# Combine train and test data
combined_data = np.concatenate((train_data, test_data), axis=0)

# Convert to DataFrame
columns = X.columns.tolist() + [f"encoded_{i}" for i in range(X_train_encode.shape[1])] + ['class']
combined_df = pd.DataFrame(combined_data, columns=columns)

# Save to CSV
combined_df.to_csv("D:/new researches/Heart disease/M4 AUTOENCODER/RES_encodedM4_statlog.csv", index=False)


# define the model
model = XGBClassifier()
# fit the model on the training set
model.fit(X_train_merged, y_train)
# make predictions on the test set
yhat = model.predict(X_test_merged)
# calculate classification accuracy
acc = accuracy_score(y_test, yhat)
print(acc)
