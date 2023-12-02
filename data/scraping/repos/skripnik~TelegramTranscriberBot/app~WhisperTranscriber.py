import time

import openai
import os
from dotenv import load_dotenv


class WhisperTranscriber:
    SUPPORTED_EXTENSIONS = ['m4a', 'mp3', 'webm', 'mp4', 'mpga', 'wav', 'mpeg', 'ogg', 'oga', 'flac']
    MAX_FILE_SIZE_MB = 25
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # Delay in seconds

    @classmethod
    def is_file_extension_supported(cls, audio_file_path) -> bool:
        _, file_ext = os.path.splitext(audio_file_path)
        return file_ext[1:] in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def is_file_size_acceptable(cls, audio_file_path) -> bool:
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        return file_size_mb <= cls.MAX_FILE_SIZE_MB

    @classmethod
    def validate_file(cls, audio_file_path) -> bool:
        if not cls.is_file_extension_supported(audio_file_path):
            return False
        if not cls.is_file_size_acceptable(audio_file_path):
            return False
        return True

    @staticmethod
    def transcribe_audio(audio_file_path) -> str:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not WhisperTranscriber.validate_file(audio_file_path):
            raise Exception("The provided file is not valid.")

        for attempt in range(WhisperTranscriber.MAX_RETRIES):
            try:
                with open(audio_file_path, "rb") as audio_file:
                    response = openai.Audio.transcribe("whisper-1", audio_file)
                return response['text']
            except openai.error.APIError as e:
                if e.http_status == 502 and attempt < WhisperTranscriber.MAX_RETRIES - 1:
                    time.sleep(WhisperTranscriber.RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"An exception occurred while trying to transcribe the audio: {e}")
