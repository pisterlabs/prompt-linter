import os
import music21
import openai

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")



def generate_chord_progression(genre ,key, time_signature, bar_length):
    # Use ChatGPT to generate chord symbols
    system_message = (
        "You are a helpful assistant that writes beautiful chord progressions. You will be given the genre, key, time signature, and the progression length in bars. You are to output each chord in the chord progressions like so: ChordRootNote ChordQuality ChordLengthInQuarterNotes. Also make sure to add a comma between each chord"
    )
    prompt = (
        f"Create a {genre}chord progression in the key of {key} with a time signature of {time_signature} and {bar_length} bars long."
    )
    user_message = (
        "C Major 4, G Major 4, A minor 4, F Major 4"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.5
    )

    chord_symbols = []
    for message in response.choices[0].message["content"].split(","):
        chord_symbols.append(str(message.strip()))
    return chord_symbols

def create_chord_from_symbol(symbol):
    print(f"Unsplit symbol is {symbol}")
    root_symbol, quality, chord_length_str = symbol.split()
    chord_length = float(chord_length_str)  # Convert the chord length to a numerical value
    print(f"Creating a chord with root note {root_symbol}, {quality} quality, and {chord_length} beats long.")
    root_note = music21.note.Note(root_symbol)
    root_midi = root_note.pitch.midi

    if quality == "Major":
        third_midi = root_midi + 4
        fifth_midi = root_midi + 7
    elif quality == "minor":
        third_midi = root_midi + 3
        fifth_midi = root_midi + 7
    else:
        print("Returned an empty chord because quality wasn't recognized")
        return music21.chord.Chord([], quarterLength=chord_length)
    # You can add more cases for other chord qualities here

    chord = music21.chord.Chord([root_midi, third_midi, fifth_midi], quarterLength=chord_length)
    return chord


def create_midi_file(chord_symbols, key, time_signature, bar_length, bpm):
    output_dir = "MIDI Files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    midi_stream = music21.stream.Score()
    part = music21.stream.Part()

    for chord_symbol in chord_symbols:
        print(f"Chord Symbol is {chord_symbol}")
        chord = create_chord_from_symbol(chord_symbol)
        part.append(chord)

    midi_stream.insert(0, part)

    # Set the tempo and time signature
    midi_stream[0].insert(0, music21.tempo.MetronomeMark(number=bpm))
    midi_stream[0].insert(0, music21.meter.TimeSignature(time_signature))

    # Save the MIDI file
    output_file = f"{output_dir}/{genre.replace('/', '-')}_chord_progression_{key}_{time_signature.replace('/', '-')}_{bpm}bpm.mid"
    midi_stream.write("midi", fp=output_file)

    print(f"MIDI file saved: {output_file}")

# Get user input for key, BPM, time signature, and bar length
genre = input("Enter the genre (e.g., Pop Rock): ")
key = input("Enter the key (e.g., Cmaj): ")
bpm = int(input("Enter the BPM: "))
time_signature = input("Enter the time signature (e.g., 4/4): ")
bar_length = int(input("Enter the bar length: "))

# Generate chord progression
chord_symbols = generate_chord_progression(genre, key, time_signature, bar_length)

print("Generated Chord Symbols:", chord_symbols)

# Create MIDI file
print("Creating MIDI file...")
create_midi_file(chord_symbols, key, time_signature, bar_length, bpm)
