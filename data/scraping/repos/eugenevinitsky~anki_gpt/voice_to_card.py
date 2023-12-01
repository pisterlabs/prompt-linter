import os
import openai
import time
import subprocess
import pvporcupine
import struct
import requests
import pyaudio
import wave

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up PyAudio
pa = pyaudio.PyAudio()

# Define the audio parameters
chunk_size = 256
sample_rate = 44100
format = pyaudio.paInt16
channels = 1

# set up pvporcupine
access_key = os.environ["PVPORCUPINE_API_KEY"]

handle = pvporcupine.create(access_key=access_key, keywords=['picovoice', 'hey barista'])
pa = pyaudio.PyAudio()

audio_stream = pa.open(
                rate=handle.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=handle.frame_length)

# set up OPEN AI
prompt = """Prompt: Your job is to take time-stamped text and convert the questions in them into question-answer format. The questions to convert will always start with the word question. If it doesn't start with the word question, don't convert it.
Do not say anything except for extracting the question and the answer!
Oh, stories, apples, hungry. Question, what year was George Washington born? Answer, 1976. Question, when did Lincoln go to the moon? Answer, 1220. I am hungry for apples. 
The moon is green. How are you doing? Well. Question, how do I ride a bycicle? Answer, using my legs.

Here's the desired outcome:

Q: What year was George Washington born?;A: 1976

Q: When did Lincoln go to the moon?;A: 1220

Q: How do I ride a bicycle?A: using my legs.

Here's some new text to process.
"""


# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the path to the file to watch
filename = "anki_speech.txt"

# Continuously check if the file exists and process its contents
try:
    while True:
        pcm = audio_stream.read(handle.frame_length)
        pcm = struct.unpack_from("h" * handle.frame_length, pcm)

        keyword_index = handle.process(pcm)

        if keyword_index >= 0:
            pa.close(audio_stream)
            print('WE SAW A KEYWORD INDEX')
            chunk = 1024      # Each chunk will consist of 1024 samples
            sample_format = pyaudio.paInt16      # 16 bits per sample
            channels = 1      # Number of audio channels
            fs = 44100        # Record at 44100 samples per second
            time_in_seconds = 10
            audio_filename = "soundsample.wav"
            
            p = pyaudio.PyAudio()  # Create an interface to PortAudio
            
            print('-----Now Recording-----')
            
            #Open a Stream with the values we just defined
            stream = p.open(format=sample_format,
                            channels = channels,
                            rate = fs,
                            frames_per_buffer = chunk,
                            input = True)
            
            frames = []  # Initialize array to store frames
            
            # Store data in chunks for 3 seconds
            for i in range(0, int(fs / chunk * time_in_seconds)):
                data = stream.read(chunk)
                frames.append(data)
            
            # Stop and close the Stream and PyAudio
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            print('-----Finished Recording-----')
            
            # Open and Set the data of the WAV file
            file = wave.open(audio_filename, 'wb')
            file.setnchannels(channels)
            file.setsampwidth(p.get_sample_size(sample_format))
            file.setframerate(fs)
            
            #Write and Close the File
            file.writeframes(b''.join(frames))
            file.close()

            # Stream audio to Whisper API
            audio_file = open(audio_filename, "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            # Save result to file
            with open(filename, "w") as f:
                f.write(transcript.text)

            print('we have finished analyzing and we are good to go')
            # Insert detection event callback here
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, "r") as f:
                    # Read the contents of the file
                    text = f.read().strip()
                    # Delete the file
                    os.remove(filename)
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt + "\n" + text + "\n" + "Q:",
                        max_tokens=40,
                        temperature=0
                        )
                    print("the API response is ", response['choices'][0]['text'])
                    with open("anki_cards.txt", "a") as f:
                        f.write("Q: " + response['choices'][0]['text']+ "\n")

            # reopen the porcupine audio stream
            audio_stream = pa.open(
                rate=handle.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=handle.frame_length)
except KeyboardInterrupt:
    # Check if the subprocess is still running
    if p.poll() is None:
        # If the subprocess is still running, terminate it
        p.terminate()
