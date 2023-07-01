import pretty_midi

# Load MIDI file into PrettyMIDI object
midi_data = pretty_midi.PrettyMIDI('../results/output2.mid')

# Retrieve list of instruments
instruments = midi_data.instruments

# Print out details about each instrument
for i, instrument in enumerate(instruments):
    print(f"Instrument {i}:")
    print(f"  - Program number: {instrument.program}")
    print(f"  - Is drum: {instrument.is_drum}")
    print(f"  - Number of notes: {len(instrument.notes)}")

# Analyze the structure of the MIDI file
print(f"Number of time signature changes: {len(midi_data.time_signature_changes)}")
print(f"Number of key signature changes: {len(midi_data.key_signature_changes)}")

# You can also retrieve the piano roll of the MIDI file
piano_roll = midi_data.get_piano_roll()

# And you can retrieve the specific notes played by an instrument
for note in instruments[0].notes:
    print(f"Note {note.pitch} starts at {note.start} and ends at {note.end}")
