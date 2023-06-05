import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2

tf.config.run_functions_eagerly(True)


# Read the CSV file
with open('../data/BEAT1R.csv', 'r') as f:
    lines = f.readlines()

# Split each line on the comma character
lines = [line.split(',') for line in lines]

# Construct the DataFrame
data = pd.DataFrame(lines)

data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace 'None' strings with actual None values
data = data.replace('None', None)

# Print the cleaned data
#print(data.head())

notes = data[data[2].str.startswith('Note')]

# Convert note events to numerical format
notes[4] = notes[4].astype("float32")  # Note pitch
notes[5] = notes[5].astype("float32")  # Note velocity

# Normalize data
notes[4] = notes[4] / 127  # MIDI notes range from 0 to 127
notes[5] = notes[5] / 127  # MIDI velocity ranges from 0 to 127

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

# Reshape the data
x_train = x_train.reshape((x_train.shape[0], sequence_length*2))

# Define the model
model = Sequential()
model.add(Dense(units=64, activation='relu',
          input_dim=sequence_length*2, kernel_regularizer=l2(0.01)))
model.add(Dense(units=2, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='sgd')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

model.save('../results/drummer.h5')