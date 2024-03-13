import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
from matplotlib import pyplot

# Load your dataset
df = pd.read_csv('D:/new researches/send/new DGA paper/new DB/NEW/DBWITHNEWFEATURES/ISOwithnewfeatures.csv')

# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']

# Number of input columns
n_inputs = X.shape[1]

# Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

# Define encoder
visible = Input(shape=(n_inputs,))
# Encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# Encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# Bottleneck
n_bottleneck = round(float(n_inputs) / 2.0)
bottleneck = Dense(n_bottleneck)(e)
# Define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# Decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# Output layer
output = Dense(n_inputs, activation='linear')(d)
# Define autoencoder model
model = Model(inputs=visible, outputs=output)

# Compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# Define the search space for Bayesian optimization
pbounds = {
    'epochs': (50, 200),
    'batch_size': (8, 32),
}

# Objective function to minimize (negative validation loss)
def objective(epochs, batch_size):
    # Ensure parameters are integers
    epochs, batch_size = int(epochs), int(batch_size)

    # Train the model
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, X_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    # Return the negative validation loss (Bayesian Optimization seeks to minimize)
    return -history.history['val_loss'][-1]

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

# Maximize the objective function (minimize negative validation loss)
optimizer.maximize(init_points=5, n_iter=10)

# Get the best parameters
best_params = optimizer.max['params']
best_epochs = int(best_params['epochs'])
best_batch_size = int(best_params['batch_size'])

# Retrain the model with the best hyperparameters
history = model.fit(X_train, X_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2, validation_data=(X_test, X_test))

# Plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_compress.png', show_shapes=True)

# Save the encoder to file
encoder.save('encoder2.h5')
