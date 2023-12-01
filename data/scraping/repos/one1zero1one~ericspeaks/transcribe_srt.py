from openai import OpenAI
import os
import sys
import glob

client = OpenAI()

def transcribe_audio(audio_file_path):
    print(f"Transcribing audio file: {audio_file_path}")
    with open(audio_file_path, "rb") as audio_file:
        # Get the transcription
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            language="ro",
            response_format="text"
        )
        # Return the transcription text directly
        return transcript

def transcribe_segments(base_name):
    # Pattern to find all segment files
    segment_pattern = f"{base_name}_*.mp3"

    # Find and process each segment
    for segment_path in glob.glob(segment_pattern):
        transcript = transcribe_audio(segment_path)
        # Create a new file with the same name as the segment but with .txt extension
        output_file_path = os.path.splitext(segment_path)[0] + ".srt"
        print(f"Writing transcript to file: {output_file_path}")
        with open(output_file_path, "w") as output_file:
            output_file.write(transcript)

# Example usage
base_name = sys.argv[1]
print(f"Processing audio segments for base name: {base_name}")
transcribe_segments(base_name)
