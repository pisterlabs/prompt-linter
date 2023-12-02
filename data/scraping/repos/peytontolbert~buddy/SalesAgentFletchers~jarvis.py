import whisper
import openai
import datetime
import sounddevice as sd
import numpy as np

FREQ = 44100  # Sample frequency
DURATION = 5  # Duration of recording in seconds



messages = [
    {"role": "system", "content": """You are an assistant named Jarvis. Only respond to help when called upon.
"""}
]





def record_audio(silence_threshold=0.05, silence_duration=1.0, min_duration=3.0):
    print('Recording')
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S")  # Changed ':' to '_'
    filepath = f"./recordings/{filename}.wav"
    min_chunks = int(FREQ * min_duration / DURATION)  # Minimum number of chunks to record

    # Continuous recording function
    with sd.InputStream(samplerate=FREQ, channels=1) as stream:
        audio_frames = []
        silent_chunks = 0
        silence_chunk_duration = int(FREQ * silence_duration / DURATION)  # Number of chunks of silence before stopping

        has_input = False  # Flag to check if there's any non-silent input
        total_chunks = 0  # Counter for total chunks recorded


        while True:
            audio_chunk, overflowed = stream.read(DURATION)
            audio_frames.append(audio_chunk)

            # Check volume of the audio chunk
            volume_norm = np.linalg.norm(audio_chunk) / len(audio_chunk)
            
            # If volume below the threshold, we consider it as silence
            if volume_norm < silence_threshold:
                if has_input:  # Only increment silent_chunks if we've had non-silent input
                    silent_chunks += 1
            else:
                silent_chunks = 0
                has_input = True  # Set the flag when we detect non-silent input
            total_chunks+=1

            # If silence for a certain duration after non-silent input, stop recording
            if silent_chunks > silence_chunk_duration and has_input and total_chunks > min_chunks:
                break

        # Save the audio
        recording = np.concatenate(audio_frames, axis=0)
        wv.write(filepath, recording, FREQ, sampwidth=2)

    return filename