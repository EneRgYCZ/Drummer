import os
import visualkeras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Flatten

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Directory containing the song files
directory = '../data/songs'

# Get a list of all the CSV files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

x_train = []
y_train = []

# Process each file
for file in files:
    with open(os.path.join(directory, file), 'r') as f:
        lines = f.readlines()

    # Split each line on the comma character
    lines = [line.split(',') for line in lines]

    # Construct the DataFrame
    data = pd.DataFrame(lines)

    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Replace 'None' strings with actual None values
    data = data.replace('None', None)

    notes = data[data[2].str.startswith('Note')]

    # Convert note events to numerical format
    notes[4] = notes[4].astype("float32")  # Note pitch
    notes[5] = notes[5].astype("float32")  # Note velocity

    # Normalize data
    notes[4] = notes[4] / 127  # MIDI notes range from 0 to 127
    notes[5] = notes[5] / 127  # MIDI velocity ranges from 0 to 127

    # Split data into sequences for training
    sequence_length = 16
    for i in range(len(notes) - sequence_length):
        x_train.append(notes.iloc[i:i+sequence_length, 4:6].values)
        y_train.append(notes.iloc[i+sequence_length, 4:6].values)

# Convert to numpy arrays and then to float32
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

# Reshape the data
x_train = x_train.reshape((x_train.shape[0], sequence_length, 2))

# Define the model
model = Sequential()
model.add(LSTM(units=64, input_shape=(sequence_length, 2), return_sequences=True, kernel_regularizer=l2(0.003)))
model.add(Flatten())
model.add(Dense(units=2, activation='relu'))

learning_rate = 0.02

early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Define the optimizer
optimizer = SGD(lr=learning_rate)

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.summary()

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[early_stop], validation_split=0.3)

# Save the model
model.save('../results/drummer2.h5')