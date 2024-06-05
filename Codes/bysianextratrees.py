import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier  # Import Extra Trees classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
import warnings

#imputed_concatenated_data_2labels
#concatenated5DB
#encodedM1_imputed_concatenated_2labels (feature generation)
#encodedM1_5DBS (feature generation)
#RES_encodedM4_imputed_concatenated_data_2labels (feature reduction with residuals)
#RESencodedM4_concatenated5DB

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

df = pd.read_csv('D:/new researches/send/Heart disease/M4 AUTOENCODER feature reduction DB/RESencodedM4_concatenated5DB.csv')

# Show class distribution
print("Class distribution in the whole dataset:")
print(df['class'].value_counts())

a = df.describe()

# Basic data preparation
X = np.array(df.drop(columns=['class']))  # Input
y = np.array(df['class'])  # Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a black_box_function specific to Extra Trees
def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    model = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Set the parameter bounds for optimization
pbounds = {
    "n_estimators": (10, 200),
    "max_depth": (1, 100),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
    "max_features": (0.1, 1.0),
}

# Create a BayesianOptimization optimizer
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=4, allow_duplicate_points=True )  # Allow duplicate points

# Perform the optimization
optimizer.maximize(init_points=20, n_iter=100)

# Get the best result
best_params = optimizer.max["params"]
best_accuracy = optimizer.max["target"]

print("Best result: Parameters={}, Accuracy={:.2f}".format(best_params, best_accuracy))
