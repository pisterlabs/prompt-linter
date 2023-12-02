# ### Import libraries

from gtts import gTTS
import math
from pydub import AudioSegment
import numpy as np
from pydub.playback import play
import librosa
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig
import psola
import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,  LLMChain
import sys


# ### Define intermediate functions

# Function to create a chord audio segment
def create_chord(frequencies, duration_ms, volume=0.5):
    sample_rate = 44100  # Standard audio sample rate
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * (duration_ms / 1000)), False)
    chord = sum([np.sin(2 * np.pi * f * t) for f in frequencies])
    chord = (chord / np.max(np.abs(chord)) * volume * 32767).astype(np.int16)  # Normalize and convert to 16-bit PCM
    return AudioSegment(chord.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

# Function to create a drum sound
def create_drum_sound(frequency, duration_ms, volume=1):
    sample_rate = 44100  # Standard audio sample rate
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * (duration_ms / 1000)), False)
    drum_sound = (volume * np.sin(2 * np.pi * frequency * t))
    
    decay_samples = int(sample_rate * duration_ms/1000)
    decay = np.linspace(1.0, 0.0, decay_samples)  # Linear decay from 1 to 0
    drum_sound[:decay_samples] *= decay  # Apply the envelope to the sine wave

    drum_sound = (drum_sound / np.max(np.abs(drum_sound)) * volume * 32767).astype(np.int16)  # Normalize and convert to 16-bit PCM
    return AudioSegment(drum_sound.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

# The next three functions focus on the auto-tune aspect
# Help was taken from https://thewolfsound.com/how-to-auto-tune-your-voice-with-python/

def corrected_note(f0, scale):
#     """Return the pitch from the stack of notes we created"""
    return librosa.midi_to_hz(midi_notes[mel_stack.pop(0)])


def get_corrected_notes(f0, scale):
#     """Map each pitch in the f0 array to the corrected note"""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = corrected_note(f0[i], scale)
    # Perform median filtering to additionally smooth the corrected pitch.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] =         sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr):
    # Set some basis parameters.
    
    #SET FRAME LENGTH AND HOP LENGTH AS BPM
    frame_length = 4500
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Pitch tracking using the PYIN algorithm.
    f0, voiced_flag, voiced_probabilities = librosa.pyin(audio,
                                                         frame_length=frame_length,
                                                         hop_length=hop_length,
                                                         sr=sr,
                                                         fmin=fmin,
                                                         fmax=fmax)
    

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = get_corrected_notes(f0,"C:maj")
    
    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)


# ### Write the main function


#Initilaise the chords, notes and drums
def initialise_music():

    global freq,midi_notes,midi_to_notes,CHORDS,BEAT

    freq = {"C0": 16.35, "C#0": 17.32, "Db0": 17.32,"D0": 18.35, "D#0": 19.45, "Eb0": 19.45,"E0": 20.60, "F0": 21.83, "F#0": 23.12, "Gb0": 23.12,"G0": 24.50, "G#0": 25.96, "Ab0": 25.96,"A0": 27.50, "A#0": 29.14, "Bb0": 29.14,"B0": 30.87,"C1": 32.70, "C#1": 34.65, "Db1": 34.65,"D1": 36.71, "D#1": 38.89, "Eb1": 38.89,"E1": 41.20, "F1": 43.65, "F#1": 46.25, "Gb1": 46.25,"G1": 49.00, "G#1": 51.91, "Ab1": 51.91,"A1": 55.00, "A#1": 58.27, "Bb1": 58.27,"B1": 61.74,"C2": 65.41, "C#2": 69.30, "Db2": 69.30,"D2": 73.42, "D#2": 77.78, "Eb2": 77.78,"E2": 82.41, "F2": 87.31, "F#2": 92.50, "Gb2": 92.50,"G2": 98.00, "G#2": 103.83, "Ab2": 103.83,"A2": 110.00, "A#2": 116.54, "Bb2": 116.54,"B2": 123.47,"C3": 130.81, "C#3": 138.59, "Db3": 138.59,"D3": 146.83, "D#3": 155.56, "Eb3": 155.56,"E3": 164.81, "F3": 174.61, "F#3": 185.00, "Gb3": 185.00,"G3": 196.00, "G#3": 207.65, "Ab3": 207.65,"A3": 220.00, "A#3": 233.08, "Bb3": 233.08,"B3": 246.94,"C4": 261.63, "C#4": 277.18, "Db4": 277.18,"D4": 293.66, "D#4": 311.13, "Eb4": 311.13,"E4": 329.63, "F4": 349.23, "F#4": 369.99, "Gb4": 369.99,"G4": 392.00, "G#4": 415.30, "Ab4": 415.30,"A4": 440.00, "A#4": 466.16, "Bb4": 466.16,"B4": 493.88,"C5": 523.25, "C#5": 554.37, "Db5": 554.37,"D5": 587.33, "D#5": 622.25, "Eb5": 622.25,"E5": 659.26, "F5": 698.46, "F#5": 739.99, "Gb5": 739.99,"G5": 783.99, "G#5": 830.61, "Ab5": 830.61,"A5": 880.00, "A#5": 932.33, "Bb5": 932.33,"B5": 987.77,"C6": 1046.50, "C#6": 1108.73, "Db6": 1108.73,"D6": 1174.66, "D#6": 1244.51, "Eb6": 1244.51,"E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "Gb6": 1479.98,"G6": 1567.98, "G#6": 1661.22, "Ab6": 1661.22,"A6": 1760.00, "A#6": 1864.66, "Bb6": 1864.66,"B6": 1975.53,"C7": 2093.00, "C#7": 2217.46, "Db7": 2217.46,"D7": 2349.32, "D#7": 2489.02, "Eb7": 2489.02,"E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "Gb7": 2959.96,"G7": 3135.96, "G#7": 3322.44, "Ab7": 3322.44,"A7": 3520.00, "A#7": 3729.31, "Bb7": 3729.31,"B7": 3951.07,"C8": 4186.01, "C#8": 4434.92, "Db8": 4434.92,"D8": 4698.63, "D#8": 4978.03, "Eb8": 4978.03,"E8": 5274.04, "F8": 5587.65, "F#8": 5919.91, "Gb8": 5919.91,"G8": 6271.93, "G#8": 6644.88, "Ab8": 6644.88}

    midi_notes = {'C0': 12, 'C#0': 13, 'Db0': 13, 'D0': 14, 'D#0': 15, 'Eb0': 15, 'E0': 16, 'F0': 17, 'F#0': 18, 'Gb0': 18, 'G0': 19, 'G#0': 20, 'Ab0': 20, 'A0': 21, 'A#0': 22, 'Bb0': 22, 'B0': 23,'C1': 24, 'C#1': 25, 'Db1': 25, 'D1': 26, 'D#1': 27, 'Eb1': 27, 'E1': 28, 'F1': 29, 'F#1': 30, 'Gb1': 30, 'G1': 31, 'G#1': 32, 'Ab1': 32, 'A1': 33, 'A#1': 34, 'Bb1': 34, 'B1': 35,'C2': 36, 'C#2': 37, 'Db2': 37, 'D2': 38, 'D#2': 39, 'Eb2': 39, 'E2': 40, 'F2': 41, 'F#2': 42, 'Gb2': 42, 'G2': 43, 'G#2': 44, 'Ab2': 44, 'A2': 45, 'A#2': 46, 'Bb2': 46, 'B2': 47,'C3': 48, 'C#3': 49, 'Db3': 49, 'D3': 50, 'D#3': 51, 'Eb3': 51, 'E3': 52, 'F3': 53, 'F#3': 54, 'Gb3': 54, 'G3': 55, 'G#3': 56, 'Ab3': 56, 'A3': 57, 'A#3': 58, 'Bb3': 58, 'B3': 59,'C4': 60, 'C#4': 61, 'Db4': 61, 'D4': 62, 'D#4': 63, 'Eb4': 63, 'E4': 64, 'F4': 65, 'F#4': 66, 'Gb4': 66, 'G4': 67, 'G#4': 68, 'Ab4': 68, 'A4': 69, 'A#4': 70, 'Bb4': 70, 'B4': 71,'C5': 72, 'C#5': 73, 'Db5': 73, 'D5': 74, 'D#5': 75, 'Eb5': 75, 'E5': 76, 'F5': 77, 'F#5': 78, 'Gb5': 78, 'G5': 79, 'G#5': 80, 'Ab5': 80, 'A5': 81, 'A#5': 82, 'Bb5': 82, 'B5': 83,'C6': 84, 'C#6': 85, 'Db6': 85, 'D6': 86, 'D#6': 87, 'Eb6': 87, 'E6': 88, 'F6': 89, 'F#6': 90, 'Gb6': 90, 'G6': 91, 'G#6': 92, 'Ab6': 92, 'A6': 93, 'A#6': 94, 'Bb6': 94, 'B6': 95,'C7': 96, 'C#7': 97, 'Db7': 97, 'D7': 98, 'D#7': 99, 'Eb7': 99, 'E7': 100, 'F7': 101, 'F#7': 102, 'Gb7': 102, 'G7': 103, 'G#7': 104, 'Ab7': 104, 'A7': 105, 'A#7': 106, 'Bb7': 106, 'B7': 107}

    midi_to_notes = {12: 'C0', 13: 'C#0', 14: 'D0', 15: 'D#0', 16: 'E0', 17: 'F0', 18: 'F#0', 19: 'G0', 20: 'G#0', 21: 'A0', 22: 'A#0', 23: 'B0',24: 'C1', 25: 'C#1', 26: 'D1', 27: 'D#1', 28: 'E1', 29: 'F1', 30: 'F#1', 31: 'G1', 32: 'G#1', 33: 'A1', 34: 'A#1', 35: 'B1',36: 'C2', 37: 'C#2', 38: 'D2', 39: 'D#2', 40: 'E2', 41: 'F2', 42: 'F#2', 43: 'G2', 44: 'G#2', 45: 'A2', 46: 'A#2', 47: 'B2',48: 'C3', 49: 'C#3', 50: 'D3', 51: 'D#3', 52: 'E3', 53: 'F3', 54: 'F#3', 55: 'G3', 56: 'G#3', 57: 'A3', 58: 'A#3', 59: 'B3',60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4', 67: 'G4', 68: 'G#4', 69: 'A4', 70: 'A#4', 71: 'B4',72: 'C5', 73: 'C#5', 74: 'D5', 75: 'D#5', 76: 'E5', 77: 'F5', 78: 'F#5', 79: 'G5', 80: 'G#5', 81: 'A5', 82: 'A#5', 83: 'B5',84: 'C6', 85: 'C#6', 86: 'D6', 87: 'D#6', 88: 'E6', 89: 'F6', 90: 'F#6', 91: 'G6', 92: 'G#6', 93: 'A6', 94: 'A#6', 95: 'B6',96: 'C7', 97: 'C#7', 98: 'D7', 99: 'D#7', 100: 'E7', 101: 'F7', 102: 'F#7', 103: 'G7', 104: 'G#7', 105: 'A7', 106: 'A#7', 107: 'B7',108: 'C8', 109: 'C#8', 110: 'D8', 111: 'D#8', 112: 'E8', 113: 'F8', 114: 'F#8', 115: 'G8', 116: 'G#8', 117: 'A8', 118: 'A#8', 119: 'B8',120: 'C9', 121: 'C#9', 122: 'D9', 123: 'D#9', 124: 'E9', 125: 'F9', 126: 'F#9', 127: 'G9'}

    CHORDS = ['Cmaj','Gmaj','Amin','Fmaj']

    BEAT = ['KSKKS','KSKKS','KSKKS','KSKKS']
    
    chord_duration_ms = 3000  # 2 seconds (4 beats)
    chord_volume = 0.5

    Cfreq = freq['C3'],freq['C4'],freq['E4'],freq['G4']
    Gfreq = freq['G3'],freq['G4'],freq['B3'],freq['D4']
    Ffreq = freq['F3'],freq['F4'],freq['A4'],freq['C4']
    Afreq = freq['A3'],freq['A4'],freq['C4'],freq['E4']

    Cmaj = create_chord(Cfreq, chord_duration_ms, chord_volume)
    Gmaj = create_chord(Gfreq, chord_duration_ms, chord_volume)
    Amin = create_chord(Afreq, chord_duration_ms, chord_volume)
    Fmaj = create_chord(Ffreq, chord_duration_ms, chord_volume)

    Cmaj.export("store/synth/Cmaj_default.mp3", format="mp3")
    Gmaj.export("store/synth/Gmaj_default.mp3", format="mp3")
    Amin.export("store/synth/Amin_default.mp3", format="mp3")
    Fmaj.export("store/synth/Fmaj_default.mp3", format="mp3")
    
    # Define drum sounds (e.g., kick and snare)
    kick = create_drum_sound(freq['C1'], 750)  # Kick drum at 80 Hz for 500 ms
    snare = create_drum_sound(freq['C3'], 750)  # Snare drum at 200 Hz for 300 ms
    kick_half = create_drum_sound(freq['C1'], 375)  # Kick drum at 80 Hz for 500 ms

    kick.export("store/drums/kick_default.mp3", format="mp3")
    snare.export("store/drums/snare_default.mp3", format="mp3")
    kick_half.export("store/drums/kickhf_default.mp3", format="mp3")

    return



# Create the text to speech segments, apply auto-tune and layer it with the music

def create_music(words,melody,chords,beat,output_path="final_song.mp3"):    

    #Split sentences and convert to text to speech for each sentence.
    SENTS = words.split('\n')

    SENTS = [i for i in SENTS if len(i)>0 ]

    for i in range(0,len(SENTS)):
#         print(SENTS[i])
        tts = gTTS(SENTS[i])
        tts.save("store/sents/" + str(i) + ".mp3")

    #Get chords.
    chordict = {}
    
    Cmaj = AudioSegment.from_mp3("store/synth/Cmaj_default.mp3")
    Gmaj = AudioSegment.from_mp3("store/synth/Gmaj_default.mp3")
    Amin = AudioSegment.from_mp3("store/synth/Amin_default.mp3")
    Fmaj = AudioSegment.from_mp3("store/synth/Fmaj_default.mp3")

    chordict['Cmaj'] = Cmaj[:3000]
    chordict['Gmaj'] = Gmaj[:3000]
    chordict['Amin'] = Amin[:3000]
    chordict['Fmaj'] = Fmaj[:3000]

    #Get drums.
    drumdict = {}

    kick = AudioSegment.from_mp3("store/drums/kick_default.mp3")
    snare = AudioSegment.from_mp3("store/drums/snare_default.mp3")
    kick_half = AudioSegment.from_mp3("store/drums/kickhf_default.mp3")

    drumdict['KSKKS'] = kick[:750] + snare[:750] + kick_half[:375] + kick_half[:375] + snare[:750]

    #Number of times to repeat melody
    reps = math.ceil(len(SENTS)/4) 

    melody = melody*reps

    beat = beat*reps

    chords = chords*reps

    #Global melody stack, that stores the melody and then applies it onto the text to speech
    global mel_stack

    for num in range(0,len(SENTS)):
        mel_stack = []

        for i in melody[num]:
            mel_stack.extend([i]*8)

        y, sr = librosa.load("store/sents/" + str(num) + ".mp3", sr=None, mono=False)

        # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
        if y.ndim > 1:
            y = y[0, :]

        pitch_corrected_y = autotune(y, sr)


        samplerate = sr; fs = 100
        t = np.linspace(0., 1., samplerate)

        data = pitch_corrected_y

        data = data.astype('float32')

        write("store/pitch_sents/" + str(num) + ".wav", samplerate, data)

    #Join all the chords, melody(vocals) and beat into one file
    num = 0

    voice_path = "store/pitch_sents/" + str(num) + ".wav"
    Vocal = AudioSegment.from_file(file = voice_path,format = "wav")

    song_part = chordict[chords[num]].overlay(Vocal)

    song_part = song_part.overlay(drumdict[beat[num]])
    
    SONG = song_part
    
    for num in range(1,len(SENTS)):

        voice_path = "store/pitch_sents/" + str(num) + ".wav"
        Vocal = AudioSegment.from_file(file = voice_path,format = "wav")

        if len(Vocal) > 3000:
            q = (len(Vocal) + 3 - (len(Vocal)%3))/3000
            Vocal = Vocal.speedup(playback_speed=q) # speed up by 2x

        song_part = chordict[chords[num]].overlay(Vocal)

        song_part = song_part.overlay(drumdict[beat[num]])

        SONG = SONG + song_part
        
    SONG.export(output_path)
    
    return


# #### Prompt for melody

def initialize_model(template, input_variables):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    
    prompt = PromptTemplate(template= template, input_variables= input_variables)

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain

def generate_melody_from_prompt(desc_user="A fun melody"):

    os.environ['OPENAI_API_KEY']= "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"

    description_1 = "A simple melody which follows the chord"

    melody_1 = """C3 C3 G3 G3 C3 C3 G3 C3
    G3 G3 D4 G3 G3 D3 D4 B3
    A3 A4 C4 C4 A3 E4 E4 C4
    F3 F3 F4 A3 A3 C4 C4 A3"""

    description_2 = """A monotonous melody"""

    melody_2 = """C3 C3 C3 C3 C3 C3 C3 C3
    G3 G3 G3 G3 G3 G3 G3 G3
    A3 A4 A4 A4 A3 A4 A4 A4
    F3 F3 F4 F3 F3 F3 F4 F3"""

    template="""You are a musician. Famous singers come to you with descriptions of the kind of song they want you to write.
    They also give you some examples of the song melodies based on the description. Only use the examples as a template, do not copy the style.
    Output only the melody of the song that you should write. do not output anything else.
    The melody consists of 32 notes, the chord changes every 8 notes and the chord progression is always Cmaj Gmaj Amin Fmaj.

    Format of the examples is:
    -- Description: <description of the song>
    -- Melody : <melody of the song>

    ----------------------------------- EXAMPLES START -----------------------------------
    -- Description 1: ```{desc_1}``` 
    -- Melody 1: ```{melody_1}``` 

    -- Description 2: ```{desc_2}``` 
    -- Melody 2: ```{melody_2}``` 
    ----------------------------------- EXAMPLES END -----------------------------------

    Note that the melody consists of 32 notes, the chord changes every 8 notes and the chord progression is always Cmaj Gmaj Amin Fmaj.

    -- Description: ```{desc_user}``` 

    -- Melody: 

    """

    input_variables = ["desc_1","desc_2","melody_1","melody_2","desc_user"] # parameters. for tempalte

    llm_chain = initialize_model(template, input_variables)

    params = {}
    params["desc_1"] = description_1
    params["desc_2"] = description_2
    params["melody_1"] = melody_1
    params["melody_2"] = melody_2
    params["desc_user"] = "A fun melody"

    model_output = llm_chain.run(params)

    return [i.split(' ') for i in model_output.split('\n')]
