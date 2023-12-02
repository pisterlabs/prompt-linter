import openai

from database_util.redis.redis_client import api_key_manager
from voice_util.asr.voice_asr import VoiceAsr


class WhisperAsr(VoiceAsr):

    def asr_voice(self, voice_file, language='zh'):
        with open(voice_file, "rb") as audio_file:
            openai.api_key = api_key_manager.get_openai_key()
            voice_text = openai.Audio.transcribe("whisper-1", audio_file,
                                                 language=language)
            return voice_text["text"]


# print(WhisperAsr().asr_voice("/Users/hamburger/Documents/AI/music/output.wav"))
