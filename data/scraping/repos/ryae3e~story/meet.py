```python
import openai_secret_manager
import openai
import os
import wave
import pydub

def transcribe_audio(file_path):
    secrets = openai_secret_manager.get_secret("whisper")
    openai.api_key = secrets["api_key"]

    # Convert mp3 file to wav as OpenAI Whisper accepts wav format
    sound = pydub.AudioSegment.from_mp3(file_path)
    wav_file_path = os.path.splitext(file_path)[0] + ".wav"
    sound.export(wav_file_path, format="wav")

    # Open the wav file
    with wave.open(wav_file_path, "rb") as wav_file:
        # Get the sample rate
        sample_rate = wav_file.getframerate()
        # Get the number of channels
        num_channels = wav_file.getnchannels()
        # Get the number of frames
        num_frames = wav_file.getnframes()
        # Read the frames
        frames = wav_file.readframes(num_frames)

    # Transcribe the audio
    response = openai.Whisper.asr(
        engine="whisper",
        audio=frames,
        sample_rate=sample_rate,
        num_channels=num_channels
    )

    # Return the transcription
    return response["choices"][0]["text"]
```