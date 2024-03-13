import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

# Load your dataset
df = pd.read_csv('D:/new researches/Heart disease/datasets/db6/statlog+heart/statlog.csv')

# Extract features and labels
X = df.drop(columns=['class'])
y = df['class']

# number of input columns
n_inputs = X.shape[1]

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# scale data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs*4)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) *6)
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(n_inputs*4)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 3
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='relu')(d) 
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# Set the resolution of the plot
pyplot.rcParams['figure.dpi'] = 2000  # Adjust the DPI value as needed

# plot the autoencoder
plot_model(model, 'autoencoder_model1.png', show_shapes=True)

# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=500, batch_size=21, verbose=2, validation_data=(X_test,X_test))

# Retrieve loss and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

# Print the final values of loss and validation loss
print("Final Loss:", loss[-1])
print("Final Validation Loss:", val_loss[-1])

# plot loss
pyplot.plot(loss, label='train')
pyplot.plot(val_loss, label='test')
pyplot.legend()
pyplot.show()

# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_model1.png', show_shapes=True)

# save the encoder to file
encoder.save('encoder3.h5')
