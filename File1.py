from keras import models, layers, optimizers
import keras._tf_keras
import numpy as np
import keras 
keras._tf_keras.keras.preprocessing.sequence.TimeseriesGenerator
# Sequence of integers from 1 to 100
# Sequence of integers from 1 to 100
data = np.array(range(1, 101))

# Reshape data to (samples, features)
data = data.reshape((100, 1))

# Split into training and validation data
x_train, x_val = data[:80], data[80:]

# Targets for training and validation data
y_train, y_val = data[1:81], data[81:]

# Adjust y_val to have the same length as x_val
y_val = np.append(y_val, 0)

# Create generators for training and validation data
train_gen = keras._tf_keras.keras.preprocessing.sequence.TimeseriesGenerator(x_train, y_train, length=1, batch_size=10)
val_gen = keras._tf_keras.keras.preprocessing.sequence.TimeseriesGenerator(x_val, y_val, length=1, batch_size=10)

# Extract actual training and validation data from generators
x_train, y_train = zip(*[(batch[0], batch[1]) for batch in train_gen])
x_val, y_val = zip(*[(batch[0], batch[1]) for batch in val_gen])

# Convert lists to arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)

# Your existing model code
model = models.Sequential()
model.add(layers.Embedding(10000, 32))
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(32, return_sequences=True))
model.add(layers.SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Assume we have some training data in x_train and y_train
# And some validation data in x_val and y_val
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(x_val, y_val))

# Evaluate the model
results = model.evaluate(x_val, y_val)
print(f'Test loss: {results[0]}, Test accuracy: {results[1]}')
