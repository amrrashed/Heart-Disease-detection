import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'heart.csv'
df = pd.read_csv(file_path)

# Basic data preparation
X = np.array(df.drop(columns=['class']))  # input
y = np.array(df['class'])  # output

# Split the data for LazyPredict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize LazyClassifier
clf = LazyClassifier()

# Fit LazyClassifier to get the list of estimators
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display available models
print("Available models from LazyPredict:")
print(models.index)

# Define scorers
scorers = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='binary'),
    'recall': make_scorer(recall_score, average='binary'),
    'roc_auc': 'roc_auc',
    'balanced_accuracy': make_scorer(balanced_accuracy_score)
}

# Perform cross-validation for each model
results = []
for model_name, model in clf.models.items():
    print(f"Performing cross-validation for {model_name}")
    scores = {}
    for metric_name, scorer in scorers.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
        scores[f'{metric_name}_mean'] = cv_scores.mean()
        scores[f'{metric_name}_std'] = cv_scores.std()
    scores['model'] = model_name
    results.append(scores)

# Convert results to DataFrame for better readability
results_df = pd.DataFrame(results)
results_df = results_df[['model'] + [col for col in results_df if col != 'model']]
results_df = results_df.sort_values(by='accuracy_mean', ascending=False)

# Display the results
print("\nCross-Validation Results:")
print(results_df)

# Save the results to a CSV file
results_df.to_csv('lazypredict_binary_classification_results.csv', index=False)
