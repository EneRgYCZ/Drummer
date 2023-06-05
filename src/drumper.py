import os
import numpy as np
from keras.models import load_model
from midiutil import MIDIFile

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

# Create a new MIDI file with one track
midi = MIDIFile(1)

# Set the instrument for the track (optional)
# 9 is the MIDI program number for a drum kit
midi.addProgramChange(track=0, channel=0, time=0, program=9)

# Add the generated notes to the MIDI file
for i, note in enumerate(generated):
    pitch, velocity = note[0]
    # MIDI note number and velocity should be integers
    pitch = int(pitch)
    velocity = int(velocity)
    # Add the note to the MIDI file
    midi.addNote(track=0, channel=0, pitch=pitch,
                 time=i, duration=1, volume=velocity)

# Write the MIDI file to disk
with open("../results/output.mid", "wb") as output_file:
    midi.writeFile(output_file)

    # Path to your SoundFont file
soundfont_path = "GeneralUser GS v1.471.sf2"

# Path to your MIDI file
midi_path = "../results/output.mid"

# Path to the output WAV file
wav_path = "../results/output.wav"

# Path to the output MP3 file
mp3_path = "../results/output.mp3"

# Use FluidSynth to convert the MIDI file to a WAV file
os.system(
    f"fluidsynth -ni {soundfont_path} {midi_path} -F {wav_path} -r 44100")

# Use LAME to convert the WAV file to an MP3 file
os.system(f"lame {wav_path} {mp3_path}")
