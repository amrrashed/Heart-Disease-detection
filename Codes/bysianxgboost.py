import numpy as np
import pandas as pd
import xgboost as xgb  # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

#imputed_concatenated_data_2labels
#concatenated5DB
#encodedM1_imputed_concatenated_2labels (feature generation)
#encodedM1_5DBS (feature generation)
#RES_encodedM4_imputed_concatenated_data_2labels (feature reduction with residuals)


# Prepare the data.
df = pd.read_csv('D:/new researches/send/Heart disease/M4 AUTOENCODER feature reduction DB/RESencodedM4_concatenated5DB.csv')

# Show class distribution
print("Class distribution in the whole dataset:")
print(df['class'].value_counts())

a = df.describe()

# Basic data preparation
X = np.array(df.drop(columns=['class']))  # Input
y = np.array(df['class'])  # Output
# Ensure class labels start from 0 if necessary
# y = y - 1  # Uncomment if class labels need adjustment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a black_box_function specific to XGBoost
def black_box_function(n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha):
    model = xgb.XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_child_weight=int(min_child_weight),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        random_state=42,
        use_label_encoder=False,  # Add this to suppress a warning
        eval_metric='logloss'     # Add this to specify evaluation metric
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Set the parameter bounds for optimization
pbounds = {
    "n_estimators": (10, 500),  # Range for n_estimators
    "max_depth": (1, 10),  # Range for max_depth
    "min_child_weight": (1, 10),  # Range for min_child_weight
    "subsample": (0.5, 1.0),  # Range for subsample
    "colsample_bytree": (0.5, 1.0),  # Range for colsample_bytree
    "gamma": (0, 1),  # Range for gamma
    "reg_alpha": (0, 1),  # Range for reg_alpha
}

# Create a BayesianOptimization optimizer
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=4)

# Perform the optimization
optimizer.maximize(init_points=20, n_iter=100)

# Get the best result
best_params = optimizer.max["params"]
best_accuracy = optimizer.max["target"]

print("Best result: Parameters={}, Accuracy={:.2f}".format(best_params, best_accuracy))
