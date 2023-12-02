import openai


def parse_audio(filename):
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file).text
    return transcription
