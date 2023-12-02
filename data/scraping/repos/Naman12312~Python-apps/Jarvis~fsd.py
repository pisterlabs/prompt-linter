import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
fs = 44100  # Sample rate
seconds = 3  # Duration of recording
print("listening...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 


import openai
openai.api_key="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"
transcript = openai.Audio.transcribe("whisper-1", open("output.wav", "rb"))

def speak(audio):
    a = gTTS(audio)
    a.save("audio.wav")
    from pydub import AudioSegment
    from pydub.playback import play

    sound = AudioSegment.from_wav('D:\\drive D\\Downloads\\Python apps\\Jarvis\\audio.wav')
    play(sound)
speak(transcript["text"])