import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.ensemble import ExtraTreesClassifier  # Change this import
from bayes_opt import BayesianOptimization

# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/datasets/db4/heart_statlog_cleveland_hungary_final.csv')

# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# load the model from file
encoder = load_model('encoder1.h5')

# encode the train data
X_train_encoded = encoder.predict(X_train)

# concatenate original and encoded data for training
X_train_merged = np.concatenate((X_train, X_train_encoded), axis=1)

# encode the test data
X_test_encoded = encoder.predict(X_test)

# concatenate original and encoded data for testing
X_test_merged = np.concatenate((X_test, X_test_encoded), axis=1)

# Bayesian Optimization function for Extra Trees parameters
def et_optimization(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Define Extra Trees model with specified parameters
    model = ExtraTreesClassifier(n_estimators=int(n_estimators),
                                 max_depth=int(max_depth),
                                 min_samples_split=int(min_samples_split),
                                 min_samples_leaf=int(min_samples_leaf),
                                 random_state=1)
    
    # Fit the model on the training set
    model.fit(X_train_merged, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_merged)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)    
    return accuracy

# Define parameter space for Bayesian Optimization
pbounds = {'n_estimators': (100, 1000),
           'max_depth': (3, 10),
           'min_samples_split': (2, 10),
           'min_samples_leaf': (1, 10)}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(f=et_optimization, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=10, n_iter=50)

# Get the best parameters
best_params = optimizer.max['params']
print("Best Parameters:", best_params)

# Train Extra Trees with the best parameters
best_model = ExtraTreesClassifier(n_estimators=int(best_params['n_estimators']),
                                  max_depth=int(best_params['max_depth']),
                                  min_samples_split=int(best_params['min_samples_split']),
                                  min_samples_leaf=int(best_params['min_samples_leaf']),
                                  random_state=1)

# Fit the model
best_model.fit(X_train_merged, y_train)

# Make predictions
y_pred = best_model.predict(X_test_merged)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
