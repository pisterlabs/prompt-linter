import os
import openai
from pydub import AudioSegment

# Transcribe MP3 with spoken words into markdown document.
# Used OpenAI Whisper to transcribe
# MP3 are split into chunk of 5min ( 25mb limit on the API )

# Replace with your OpenAI API key
API_KEY = ""
openai.api_key = API_KEY

def transcribe_audio(file_path):
    print(f"Transcribing with openai {file_path}...")
    with open(file_path, "rb") as audio_file:
        response = openai.Audio.translate("whisper-1", audio_file)
        transcript = response['text']
    return transcript

def split_and_transcribe(file_path, segment_duration_ms=300000):
    audio = AudioSegment.from_mp3(file_path)
    audio_length_ms = len(audio)
    transcriptions = []
    
    for start_ms in range(0, audio_length_ms, segment_duration_ms):
        print(f"Transcribing segment {start_ms} to {start_ms + segment_duration_ms}...")
        end_ms = min(start_ms + segment_duration_ms, audio_length_ms)
        segment = audio[start_ms:end_ms]
        
        segment_file_path = f"temp_{start_ms}_{end_ms}.mp3"
        segment.export(segment_file_path, format="mp3")
        
        segment_transcription = transcribe_audio(segment_file_path)
        transcriptions.append(segment_transcription)
        
        os.remove(segment_file_path)
    
    return " ".join(transcriptions)

def main():
    directory = os.getcwd()
    for filename in os.listdir(directory):
        if filename.endswith(".mp3"):
            print(f"Transcribing {filename}...")
            transcription = split_and_transcribe(filename)
            output_filename = f"{os.path.splitext(filename)[0]}.md"
            with open(output_filename, "w") as output_file:
                output_file.write(transcription)
            print(f"Transcription saved in {output_filename}")

if __name__ == "__main__":
    main()
