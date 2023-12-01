import sounddevice as sd
import numpy as np
import queue
import io
from openai import OpenAI
import wavio
import subprocess


# Parameters
sample_rate = 16000  # Sample rate in Hz
silence_threshold = 0.001  # Try increasing this value
chunk_size = 1024  # Each chunk will have 1024 samples
silence_duration = 1.9  # Number of seconds of silence before stopping
chunks_per_second = sample_rate // chunk_size

# Initialize queue to store audio chunks
audio_queue = queue.Queue()

# Callback function to capture audio
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Initialize OpenAI client
client = OpenAI()

def record_and_transcribe():
    # Start streaming audio
    with sd.InputStream(callback=audio_callback, dtype='float32', channels=1, samplerate=sample_rate, blocksize=chunk_size):
        audio_data = []
        silent_chunks = 0  # Count consecutive silent chunks
        print("Recording... Speak now")

        while True:
            data = audio_queue.get()
            avg_volume = np.mean(np.abs(data))
            #print(f"Chunk volume: {avg_volume}")  # Debugging line

            if avg_volume < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_duration * chunks_per_second:
                    print("Silence detected, processing audio...")
                    break
            else:
                silent_chunks = 0
            audio_data.append(data)

    if audio_data:
        audio_data = np.concatenate(audio_data, axis=0)

        # Normalize audio data
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        # Save audio data to a WAV file
        wav_filename = 'recorded_audio.wav'
        wavio.write(wav_filename, audio_data, sample_rate, sampwidth=2)

        with open(wav_filename, 'rb') as audio_file:
            # Transcribe the audio
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

            transcribed_text = transcript.text  # Extract the text from the Transcription object

            print("Transcription:", transcribed_text)
            #cmd_command = f"cmd /c python gpt.py \"{transcribed_text}\""
            #cmd_command = f"cmd /c python gpti.py \"{transcribed_text}\""
            subprocess.run(["python", "gpti.py", transcribed_text])
            subprocess.run(["python", "gpt.py", transcribed_text])
    else:
        print("No audio data recorded.")

# Loop to continuously record and transcribe
while True:
    try:
        wav_filename = record_and_transcribe()
    except KeyboardInterrupt:
        print("\nRecording and transcription stopped")
        break
