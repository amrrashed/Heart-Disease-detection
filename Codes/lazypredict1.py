import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# load the dataset
df = pd.read_csv('/content/encodedM2_imputed_clevelanddata_2labels.csv')
#df['class'] = df['class'].apply(lambda x: 1 if x != 0 else x)
# Save the modified dataset
#df.to_csv('imputed_vadata_2labels.csv', index=False)

#df = df.drop(columns=['fr11', 'fr12', 'fr13'])
# Plot class distribution histogram
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='class', data=df)
plt.title('Class Distribution Histogram ')
plt.xlabel('Class')
plt.ylabel('Count')

# Add numbers to the head of each column
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

# Show dataset summary
dataset_summary = df.describe(include='all')
print("Dataset Summary:")
print(dataset_summary)

# Basic data preparation
X = np.array(df.drop(['class'], 1))  # input
y = np.array(df['class'])  # output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

clf = LazyClassifier(verbose=-1, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Create a DataFrame with model names and all results
results_df = pd.DataFrame(models, columns=['Model', 'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score', 'Time Taken'])

print(results_df)

# Save the results to a CSV file
results_df.to_csv('/content/lazypredict_concatenated_data.csv', index=False)


