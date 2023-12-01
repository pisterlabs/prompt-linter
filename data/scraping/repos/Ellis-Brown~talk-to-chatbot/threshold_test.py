
from time import sleep
import sounddevice as sd
import numpy
from scipy.io.wavfile import write
import datetime
import openai 
import os
import requests
from playsound import playsound

audio_input_sample_rate = 44100  
period = 1.0
# Test the microphone's threshold. Set the min threshold in your codebase
# equal to the sound where it is quiet + 10

print("Beginning 5 second sound test")
for i in range(5):
        # Record the audio
    myrecording = sd.rec(int(period * audio_input_sample_rate), 
                            samplerate=audio_input_sample_rate, 
                            channels=1)
    
    sd.wait() # Wait for the conversation to end
    
    # Check if they stopped talking
    sound_level = sum(abs(myrecording)) / period
    print("Sound Level:", sound_level)

    