import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from bayes_opt import BayesianOptimization


# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/datasets/db4/heart_statlog_cleveland_hungary_final.csv')

# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM (number of features for each time step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# One-hot encode the target labels
num_classes = len(set(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define the CNN + LSTM model
def create_model(learning_rate, num_filters, kernel_size, lstm_units, dense_units):
    model = Sequential()
    model.add(Conv1D(filters=int(num_filters), kernel_size=int(kernel_size), activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(int(lstm_units)))
    model.add(Dense(int(dense_units), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Define the optimization bounds
pbounds = {
    'learning_rate': (0.001, 0.009),
    'num_filters': (3, 10),
    'kernel_size': (1, 10),
    'lstm_units': (60, 200),
    'dense_units': (20, 160)
}

# Bayesian optimization function
def cnn_lstm_hyperopt(learning_rate, num_filters, kernel_size, lstm_units, dense_units):
    model = create_model(learning_rate, num_filters, kernel_size, lstm_units, dense_units)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    _, val_acc = model.evaluate(X_test, y_test, verbose=0)
    return val_acc

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=cnn_lstm_hyperopt, pbounds=pbounds, verbose=2, random_state=42)
optimizer.maximize(init_points=10, n_iter=20)

# Get the best hyperparameters
best_hyperparameters = optimizer.max
print("Best Hyperparameters:", best_hyperparameters)


# import matplotlib.pyplot as plt

# # ...

# # Define the CNN model without the Bayesian optimization part
# def create_cnn_model():
#     model = Sequential()
#     model.add(Conv1D(filters=8, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
#     model.add(LSTM(153))
#     model.add(Dense(91, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
    
#     optimizer = keras.optimizers.Adam(learning_rate=0.004)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return model

# # Visualize the CNN model
# def visualize_cnn_model():
#     model = create_cnn_model()
#     keras.utils.plot_model(model, to_file='cnn_model.png', show_shapes=True)

# # Display the model blocks
# visualize_cnn_model()

# # Show the plots
# plt.show()





