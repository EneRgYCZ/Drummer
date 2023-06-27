import os
import mido
import random
import numpy as np
from midiutil import MIDIFile
from keras.models import load_model


def preprocess_midi(midi_file):
    # Load the MIDI file
    mid = mido.MidiFile(midi_file)

    # Initialize an empty list to hold the notes and velocities
    notes = []
    velocities = []

    # Iterate over all the messages in the MIDI file
    for msg in mid:
        # Check if the message is a 'note_on' message
        if msg.type == 'note_on':
            # Append the note and velocity to their respective lists
            notes.append(msg.note)
            velocities.append(msg.velocity)

    # Normalize the notes and velocities
    notes = np.array(notes) / 127.0
    velocities = np.array(velocities) / 127.0

    # Stack the notes and velocities together
    preprocessed_data = np.column_stack((notes, velocities))

    return preprocessed_data

# Load the model
model = load_model('../results/drummer.h5')

# Create a seed sequence
midi_data = mido.MidiFile('../data/midi_songs/Songs/All_Star.mid')

# Preprocess the MIDI data
preprocessed_data = preprocess_midi("../data/midi_songs/Songs/Its_My_Life.mid")

# Create the seed sequence
seed = preprocessed_data[:16]

# Reshape the seed sequence to match the input shape of your model
seed = seed.reshape(1, 16, 2)

print (seed)

# Define the number of steps to generate
num_steps = 50

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
midi.addProgramChange(0, 9, 1, 9)

beat_time = [2, 3]

# Add the generated notes to the MIDI file
for i, note in enumerate(generated):
    pitch, velocity = note[0]
    # MIDI note number and velocity should be integers
    pitch = int(pitch)
    velocity = int(velocity) + 20
    print (pitch)
    print (velocity)
    beat = random.choice(beat_time)
    print (beat)
    print ()
    # Add the note to the MIDI file
    midi.addNote(track=0, channel=9, pitch=pitch,
                 time=i / 3, duration=1, volume=velocity)

# Write the MIDI file to disk
with open("../results/output2.mid", "wb") as output_file:
    midi.writeFile(output_file)

    # Path to your SoundFont file
soundfont_path = "da.sf2"

# Path to your MIDI file
midi_path = "../results/output2.mid"

# Path to the output WAV file
wav_path = "../results/output2.wav"

# Path to the output MP3 file
mp3_path = "../results/output2.mp3"

# Use FluidSynth to convert the MIDI file to a WAV file
os.system(
    f"fluidsynth -ni {soundfont_path} {midi_path} -F {wav_path} -r 44100")

# Use LAME to convert the WAV file to an MP3 file
os.system(f"lame {wav_path} {mp3_path}")
