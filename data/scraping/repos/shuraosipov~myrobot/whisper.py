
import openai

async def transcribe_audio(file_path) -> str:
    """ Transcribe the audio file """
    audio_file= open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="text")
    return transcript



