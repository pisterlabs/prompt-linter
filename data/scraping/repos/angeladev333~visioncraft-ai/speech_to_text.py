import openai

openai.api_key = "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" 

def transcribe_text(file_path):
    audio_file=open(file_path, 'rb')
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)

transcribe_text("server\\test-audio-file.mp3")