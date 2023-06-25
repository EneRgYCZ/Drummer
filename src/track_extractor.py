import mido

# Load the MIDI file
midi = mido.MidiFile('../data/1.mid')

# Create a new MIDI file for the drum track
drum_midi = mido.MidiFile()

# Iterate over the tracks in the original MIDI file
for track in midi.tracks:
     # Check if the track name is "Drumkit"
     for msg in track:
          if msg.type == 'track_name' and msg.name == 'Drumkit':
               # Add the track to the new MIDI file
               drum_midi.tracks.append(track)
               break

# Save the new MIDI file
drum_midi.save('../data/1.mid')