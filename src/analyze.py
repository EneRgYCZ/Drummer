import os
import pretty_midi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_midi(midi_data):
     # Initialize counters
     num_notes = 0
     total_duration = 0
     total_velocity = 0
     pitches = set()

     # Iterate over all instruments and notes
     for instrument in midi_data.instruments:
          for note in instrument.notes:
               num_notes += 1
               total_duration += note.end - note.start
               total_velocity += note.velocity
               pitches.add(note.pitch)

     # Calculate averages
     avg_duration = total_duration / num_notes if num_notes else 0
     avg_velocity = total_velocity / num_notes if num_notes else 0

     # Return the results
     return {
          'number of instruments': len(midi_data.instruments),
          'number of notes': num_notes,
          'number of unique pitches': len(pitches),
          'average note duration': avg_duration,
          'avereage note velocity': avg_velocity
     }

def analyze_all_midis(directory, generated_midi):
     midi_files = os.listdir(directory)
     analyses = []

     # Analyze the generated MIDI
     midi_data = pretty_midi.PrettyMIDI(generated_midi)
     analysis = analyze_midi(midi_data)
     analysis['Name of the song'] = 'Generated Song'
     analyses.append(analysis)

     # Analyze each MIDI file in the directory
     for midi_file in midi_files:
          print(f'Analyzing {midi_file}...')  # Add this line
          try:
               midi_data = pretty_midi.PrettyMIDI(os.path.join(directory, midi_file))
               analysis = analyze_midi(midi_data)
               analysis['Name of the song'] = midi_file
               analyses.append(analysis)
          except Exception as e:
               print(f"Error while analyzing {midi_file}: {e}")
     return analyses


def plot_analyses(analyses):
     # Set global font size
     plt.rcParams.update({'font.size': 16})

     df = pd.DataFrame(analyses)
     df['Name of the song'] = df['Name of the song'].str.replace('_', ' ')  # Replace underscores with spaces
     df['Name of the song'] = df['Name of the song'].str.replace('.mid', '')  # Replace underscores with spaces

     variables = ['number of instruments', 'number of notes', 'number of unique pitches', 'average note duration', 'avereage note velocity']

     for var in variables:
          plt.figure(figsize=(10, 13))
          # Create a list of colors, with a different color for the generated song
          colors = ['blue' if song == 'Generated Song' else 'gray' for song in df['Name of the song']]
          sns.barplot(x='Name of the song', y=var, data=df, palette=colors)
          plt.title(f'Comparison of {var} between songs')
          plt.xticks(rotation=90)
          plt.tight_layout()
          plt.show()


# Directory containing your MIDI files
directory = '../data/midi_songs/Songs'
# Path to the generated MIDI file
generated_midi = '../results/output3.mid'

analyses = analyze_all_midis(directory, generated_midi)
plot_analyses(analyses)
