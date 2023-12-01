import json
from openai_wrapper import OpenAIWrapper
from voicevox import Voicevox
from pydub import AudioSegment
from pydub.playback import play
import os


class VoiceSynthesizer:
    def __init__(self, config, name, config_file="voice_synthesizer.json"):
        self.openai = None
        self.name = name
        self.config = config
        self.profile = None
        self.synth_type = None
        self.load(config_file)

    def load(self, config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                self.profile = json.load(file)[self.name]
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")

        self.synth_type = self.profile["type"]
        if self.synth_type == "openai":
            self.openai = OpenAIWrapper(self.config.openai_api_key)

    def synthesize(self, text, filename):
        if self.synth_type == "openai":
            self.openai.synthesize(text, filename, self.profile["model"], self.profile["voice"])
        elif self.synth_type == "voicebox":
            voicevox = Voicevox(self.profie["host"], self.profile["port"])
            voicevox.synthesize(text, filename)

        self.play_audio(filename)

    def play_audio(self,file_path):
        # ファイルの拡張子を取得
        file_ext = os.path.splitext(file_path)[1].lower()

        # 拡張子に基づいて適切なフォーマットで読み込む
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(file_path, "flac")
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        else:
            raise Exception("Unsupported file format")

        # オーディオを再生
        play(audio)
