import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif  # annova
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

# Load the data
df = pd.read_csv('D:/new researches/Heart disease/datasets/db4/encodedM1_5DBS.csv')

# Basic data preparation
X = df.drop(['class'], axis=1)  # Input features
y = df['class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to perform feature selection using SelectKBest with mutual_info_classif
def select_features(X_train, y_train, X_test, k_best):
    selector = SelectKBest(f_classif, k=int(k_best))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support(indices=True)]  # Get the names of selected features
    return X_train_selected, X_test_selected, selected_features

# Define a black_box_function specific to Random Forest with feature selection
def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_samples, max_leaf_nodes, k_best):
    # Feature selection using SelectKBest with mutual_info_classif
    X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test, k_best)

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        max_samples=max_samples,
        max_leaf_nodes=int(max_leaf_nodes),
        random_state=42,
    )
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Set the parameter bounds for optimization, including k_best for feature selection
pbounds = {
    "n_estimators": (10, 500),
    "max_depth": (1, 100),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 20),
    "max_features": (0.1, 1.0),
    "max_samples": (0.1, 1.0),
    "max_leaf_nodes": (10, 1000),
    "k_best": (1, 66),  # Adjust this range to match the number of features in your dataset
}

# Create a BayesianOptimization optimizer
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=4)

# Perform the optimization
optimizer.maximize(init_points=20, n_iter=100)

# Get the best result
best_params = optimizer.max["params"]
best_accuracy = optimizer.max["target"]
best_k_best = int(best_params["k_best"])  # Retrieve the best k_best value

# Perform feature selection again to get the names of selected features
X_train_selected, X_test_selected, selected_features = select_features(X_train, y_train, X_test, best_k_best)

print("Best result: Parameters={}, Accuracy={:.2f}".format(best_params, best_accuracy))
print("Selected features:", selected_features)

# Create a DataFrame with selected features and add the 'class' column
selected_df = pd.DataFrame(data=X, columns=selected_features)
selected_df['class'] = y

# Save the DataFrame to a new CSV file
selected_df.to_csv('selected_features_imputed_vadata_2labels.csv', index=False)