import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt

from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten

tf.config.run_functions_eagerly(True)

tf.data.experimental.enable_debug_mode()

# Directory containing the song files
directory = '../data/songs'

# Initialize lists to hold all training data
all_x_train = []
all_y_train = []

# Process each song file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Read the CSV file
        with open(os.path.join(directory, filename), 'r') as f:
            lines = f.readlines()

        # Split each line on the comma character
        lines = [line.split(',') for line in lines]

        # Construct the DataFrame
        data = pd.DataFrame(lines)

        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Replace 'None' strings with actual None values
        data = data.replace('None', None)

        notes = data[data[2].str.startswith('Note')].copy()

        # Convert note events to numerical format
        notes.loc[:, 4] = notes.loc[:, 4].astype("float32")  # Note pitch
        notes.loc[:, 5] = notes.loc[:, 5].astype("float32")  # Note velocity

        # Normalize data
        notes.loc[:, 4] = notes.loc[:, 4] / 127  # MIDI notes range from 0 to 127
        notes.loc[:, 5] = notes.loc[:, 5] / 127  # MIDI velocity ranges from 0 to 127

        # Split data into sequences for training
        sequence_length = 16
        x_train = []
        y_train = []
        for i in range(len(notes) - sequence_length):
            x_train.append(notes.iloc[i:i+sequence_length, 4:6].values)
            y_train.append(notes.iloc[i+sequence_length, 4:6].values)

        # Convert to numpy arrays and then to float32
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # Reshape the data
        x_train = x_train.reshape((x_train.shape[0], sequence_length*2))

        # Add the training data for this song to the overall training data
        all_x_train.append(x_train)
        all_y_train.append(y_train)

# Concatenate all training data
all_x_train = np.concatenate(all_x_train)
all_y_train = np.concatenate(all_y_train)

# Define the model
model = Sequential()
model.add(LSTM(units=64, input_shape=(sequence_length, 2), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.01)))
model.add(Flatten())
model.add(Dense(units=2, activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='sgd')

model.summary()

# Train the model
history = model.fit(all_x_train.reshape(
    (all_x_train.shape[0], sequence_length, 2)), all_y_train, epochs=100, batch_size=32)

plt.figure(figsize=(12,6))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

model.save('/results/drummer.h5')
