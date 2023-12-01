import openai

def transcribe_audio(audio_file_path):
    audio = open(audio_file_path,"rb")
    transcript = openai.Audio.transcribe("whisper-1",audio,no_speech_threshold=0.3)
    return transcript['text']