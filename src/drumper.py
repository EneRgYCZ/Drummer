import numpy as np
from keras.models import load_model

# Load the model
model = load_model('../results/drummer.h5')

# Create a seed sequence
seed = np.random.rand(1, 32)

# Define the number of steps to generate
num_steps = 100

# Create a list to hold the generated sequence
generated = []

# Generate drum hits
for _ in range(num_steps):
    # Predict the next hit
    next_hit = model.predict(seed)

    # Post-process the prediction
    next_hit = (next_hit * 127).astype(int)

    # Add the predicted hit to the generated sequence
    generated.append(next_hit)

    # Update the seed sequence
    seed = np.roll(seed, -2)
    seed[0, -2:] = next_hit / 127

# Convert the generated sequence to a numpy array
generated = np.array(generated)

# Print the generated sequence
print(generated)
