import os
import openai
from midiutil import MIDIFile
import pygame

# Set up the OpenAI API
openai.api_key = "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"

system_message = {
    "role": "system",
    "content": (
        "You are a music-generating assistant. When prompted, produce a musical response in the following format:"
        "'Sequence: [musical elements], Tempo: [BPM], Duration: [Note Duration]'."
        "Ensure the sequence only uses the provided musical elements and parameters."
        "Here is a sample of an appropriate response: "
        "Sequence: [C, D, E, F, G, A, B, C, D, E, F, E, D, C, B, A, G, F, E, D, C, B, A, G, F, E, D, C, A#, A, G, F, E, D, C#, C, B, A, G, F, E, D, C, B, A, G, F, E, D, C], Tempo: 150 BPM, Duration: Eighth notes"
        "MAKE SURE ALL RESPONSES ARE AS SUCH, DO NOT DEVIATE FROM THIS FORMAT EVER!"
    )
}

def get_music_from_openai(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return completion.choices[0].message.content

def parse_ai_response(response):
    # Extract sequence
    start_index = response.find('[')
    end_index = response.find(']')
    sequence = response[start_index+1:end_index].split(", ")
    
    # Extract tempo
    tempo_str = response.split("Tempo: ")[1]
    tempo = int(tempo_str.split(" ")[0])  # Extract the integer value
    
    # Extract duration (this assumes a single word description for now)
    duration_str = response.split("Duration: ")[1]
    duration = duration_str.split(" ")[0]  # Extract the word
    
    return sequence, tempo, duration

def create_midi_from_sequence(sequence, filename="output.mid", tempo=120, duration="Quarter"):
    # Create a new MIDI file with 1 track
    midi = MIDIFile(1)

    # Define some constants
    track = 0
    time = 0
    
    # Convert duration string to a numeric value
    if duration == "Whole":
        duration_value = 4
    elif duration == "Half":
        duration_value = 2
    elif duration == "Eighth":
        duration_value = 0.5
    else:  # Default to "Quarter"
        duration_value = 1
        
    volume = 100  # Volume, from 0-127

    midi.addTempo(track, time, tempo)

    note_mapping = {
        # Notes
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63, "E": 64, 
    "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68, 
    "A": 69, "A#": 70, "Bb": 70, "B": 71,

    # Major Chords
    "Cmaj": [60, 64, 67], "C#maj": [61, 65, 68], "Dbmaj": [61, 65, 68],
    "Dmaj": [62, 66, 69], "D#maj": [63, 67, 70], "Ebmaj": [63, 67, 70],
    "Emaj": [64, 68, 71], "Fmaj": [65, 69, 72], "F#maj": [66, 70, 73],
    "Gbmaj": [66, 70, 73], "Gmaj": [67, 71, 74], "G#maj": [68, 72, 75], 
    "Abmaj": [68, 72, 75], "Amaj": [69, 73, 76], "A#maj": [70, 74, 77],
    "Bbmaj": [70, 74, 77], "Bmaj": [71, 75, 78],

    # Minor Chords
    "Cmin": [60, 63, 67], "C#min": [61, 64, 68], "Dbmin": [61, 64, 68],
    "Dmin": [62, 65, 69], "D#min": [63, 66, 70], "Ebmin": [63, 66, 70],
    "Emin": [64, 67, 71], "Fmin": [65, 68, 72], "F#min": [66, 69, 73],
    "Gbmin": [66, 69, 73], "Gmin": [67, 70, 74], "G#min": [68, 71, 75],
    "Abmin": [68, 71, 75], "Amin": [69, 72, 76], "A#min": [70, 73, 77],
    "Bbmin": [70, 73, 77], "Bmin": [71, 74, 78],

    # Seventh Chords
    "C7": [60, 64, 67, 70], "C#7": [61, 65, 68, 71], "Db7": [61, 65, 68, 71],
    "D7": [62, 66, 69, 72], "D#7": [63, 67, 70, 73], "Eb7": [63, 67, 70, 73],
    "E7": [64, 68, 71, 74], "F7": [65, 69, 72, 75], "F#7": [66, 70, 73, 76],
    "Gb7": [66, 70, 73, 76], "G7": [67, 71, 74, 77], "G#7": [68, 72, 75, 78],
    "Ab7": [68, 72, 75, 78], "A7": [69, 73, 76, 79], "A#7": [70, 74, 77, 80],
    "Bb7": [70, 74, 77, 80], "B7": [71, 75, 78, 81],

    # Major Seventh Chords
    "Cmaj7": [60, 64, 67, 71], "C#maj7": [61, 65, 68, 72], "Dbmaj7": [61, 65, 68, 72],
    "Dmaj7": [62, 66, 69, 73], "D#maj7": [63, 67, 70, 74], "Ebmaj7": [63, 67, 70, 74],
    "Emaj7": [64, 68, 71, 75], "Fmaj7": [65, 69, 72, 76], "F#maj7": [66, 70, 73, 77],
    "Gbmaj7": [66, 70, 73, 77], "Gmaj7": [67, 71, 74, 78], "G#maj7": [68, 72, 75, 79],
    "Abmaj7": [68, 72, 75, 79], "Amaj7": [69, 73, 76, 80], "A#maj7": [70, 74, 77, 81],
    "Bbmaj7": [70, 74, 77, 81], "Bmaj7": [71, 75, 78, 82],

    # Minor Seventh Chords
    "Cmin7": [60, 63, 67, 70], "C#min7": [61, 64, 68, 71], "Dbmin7": [61, 64, 68, 71],
    "Dmin7": [62, 65, 69, 72], "D#min7": [63, 66, 70, 73], "Ebmin7": [63, 66, 70, 73],
    "Emin7": [64, 67, 71, 74], "Fmin7": [65, 68, 72, 75], "F#min7": [66, 69, 73, 76],
    "Gbmin7": [66, 69, 73, 76], "Gmin7": [67, 70, 74, 77], "G#min7": [68, 71, 75, 78],
    "Abmin7": [68, 71, 75, 78], "Amin7": [69, 72, 76, 79], "A#min7": [70, 73, 77, 80],
    "Bbmin7": [70, 73, 77, 80], "Bmin7": [71, 74, 78, 81],

        # Diminished Chords
    "Cdim": [60, 63, 66], "C#dim": [61, 64, 67], "Dbdim": [61, 64, 67],
    "Ddim": [62, 65, 68], "D#dim": [63, 66, 69], "Ebdim": [63, 66, 69],
    "Edim": [64, 67, 70], "Fdim": [65, 68, 71], "F#dim": [66, 69, 72],
    "Gbdim": [66, 69, 72], "Gdim": [67, 70, 73], "G#dim": [68, 71, 74],
    "Abdim": [68, 71, 74], "Adim": [69, 72, 75], "A#dim": [70, 73, 76],
    "Bbdim": [70, 73, 76], "Bdim": [71, 74, 77],

    # Augmented Chords
    "Caug": [60, 64, 68], "C#aug": [61, 65, 69], "Dbaug": [61, 65, 69],
    "Daug": [62, 66, 70], "D#aug": [63, 67, 71], "Ebaug": [63, 67, 71],
    "Eaug": [64, 68, 72], "Faug": [65, 69, 73], "F#aug": [66, 70, 74],
    "Gbaug": [66, 70, 74], "Gaug": [67, 71, 75], "G#aug": [68, 72, 76],
    "Abaug": [68, 72, 76], "Aaug": [69, 73, 77], "A#aug": [70, 74, 78],
    "Bbaug": [70, 74, 78], "Baug": [71, 75, 79],

    # Diminished Seventh Chords
    "Cdim7": [60, 63, 66, 69], "C#dim7": [61, 64, 67, 70], "Dbdim7": [61, 64, 67, 70],
    "Ddim7": [62, 65, 68, 71], "D#dim7": [63, 66, 69, 72], "Ebdim7": [63, 66, 69, 72],
    "Edim7": [64, 67, 70, 73], "Fdim7": [65, 68, 71, 74], "F#dim7": [66, 69, 72, 75],
    "Gbdim7": [66, 69, 72, 75], "Gdim7": [67, 70, 73, 76], "G#dim7": [68, 71, 74, 77],
    "Abdim7": [68, 71, 74, 77], "Adim7": [69, 72, 75, 78], "A#dim7": [70, 73, 76, 79],
    "Bbdim7": [70, 73, 76, 79], "Bdim7": [71, 74, 77, 80],

    # Augmented Seventh Chords
    "Caug7": [60, 64, 68, 72], "C#aug7": [61, 65, 69, 73], "Dbaug7": [61, 65, 69, 73],
    "Daug7": [62, 66, 70, 74], "D#aug7": [63, 67, 71, 75], "Ebaug7": [63, 67, 71, 75],
    "Eaug7": [64, 68, 72, 76], "Faug7": [65, 69, 73, 77], "F#aug7": [66, 70, 74, 78],
    "Gbaug7": [66, 70, 74, 78], "Gaug7": [67, 71, 75, 79], "G#aug7": [68, 72, 76, 80],
    "Abaug7": [68, 72, 76, 80], "Aaug7": [69, 73, 77, 81], "A#aug7": [70, 74, 78, 82],
    "Bbaug7": [70, 74, 78, 82], "Baug7": [71, 75, 79, 83],

    # Suspended 2nd Chords
    "Csus2": [60, 62, 67], "C#sus2": [61, 63, 68], "Dbsus2": [61, 63, 68],
    "Dsus2": [62, 64, 69], "D#sus2": [63, 65, 70], "Ebsus2": [63, 65, 70],
    "Esus2": [64, 66, 71], "Fsus2": [65, 67, 72], "F#sus2": [66, 68, 73],
    "Gbsus2": [66, 68, 73], "Gsus2": [67, 69, 74], "G#sus2": [68, 70, 75],
    "Absus2": [68, 70, 75], "Asus2": [69, 71, 76], "A#sus2": [70, 72, 77],
    "Bbsus2": [70, 72, 77], "Bsus2": [71, 73, 78],

    "Csus4": [60, 65, 67], "C#sus4": [61, 66, 68], "Dbsus4": [61, 66, 68],
    "Dsus4": [62, 67, 69], "D#sus4": [63, 68, 70], "Ebsus4": [63, 68, 70],
    "Esus4": [64, 69, 71], "Fsus4": [65, 70, 72], "F#sus4": [66, 71, 73], "Gbsus4": [66, 71, 73],
    "Gsus4": [67, 72, 74], "G#sus4": [68, 73, 75], "Absus4": [68, 73, 75],
    "Asus4": [69, 74, 76], "A#sus4": [70, 75, 77], "Bbsus4": [70, 75, 77],
    "Bsus4": [71, 76, 78]
    }

    # Add notes to the MIDI file
    for note_or_chord in sequence:
        if note_or_chord in note_mapping:
            notes = note_mapping[note_or_chord]
            if isinstance(notes, list):  # It's a chord
                for note in notes:
                    midi.addNote(track, 0, note, time, duration_value, volume)
            else:  # It's a single note
                midi.addNote(track, 0, notes, time, duration_value, volume)
            time += duration_value

    # Write the MIDI file
    with open(filename, "wb") as f:
        midi.writeFile(f)

def play_midi(filename):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
    messages = [system_message]
    user_message_content = input("What kind of music do you want to hear? Describe the mood, tempo, or any other preferences. ")
    messages.append({"role": "user", "content": user_message_content})

    while True:
        response = get_music_from_openai(messages)
        print(f"AI Response: {response}")

        sequence, tempo, duration = parse_ai_response(response)
        if sequence is None:
            print("Could not generate music from the response. Try again.")
            continue
        
        print("\nGenerated Music Sequence:\n", sequence)
        print("Tempo:", tempo)
        print("Duration:", duration)
        
        create_midi_from_sequence(sequence, "output.mid", tempo, duration)
        play_midi("output.mid")
        
        feedback = input("\nDo you want to change or add something? (Enter 'no' to save and exit) ")
        
        if feedback.lower() == 'no':
            with open("generated_music_sequence.txt", "w") as f:
                f.write(response)
            print("Music sequence saved to 'generated_music_sequence.txt'.")
            break
        else:
            messages.append({"role": "user", "content": feedback})

if __name__ == "__main__":
    main()
