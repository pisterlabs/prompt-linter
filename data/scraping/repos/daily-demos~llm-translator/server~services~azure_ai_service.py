from services.ai_service import AIService
import openai
import requests
import os
import io

# See .env.example for Azure configuration needed
from azure.cognitiveservices.speech import SpeechSynthesizer, SpeechConfig, ResultReason, CancellationReason


class AzureAIService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speech_key = os.getenv("AZURE_SPEECH_SERVICE_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_SERVICE_REGION")

        self.speech_config = SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region)
        # self.speech_config.speech_synthesis_voice_name='en-US-JennyMultilingualV2Neural'

        self.speech_synthesizer = SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None)

        self.languages = {
            "english": {
                "lang": "en-US",
                "voices": {
                    "male": [
                        "en-US-GuyNeural",
                        "en-US-DavisNeural"],
                    "female": [
                        "en-US-JennyNeural",
                        "en-US-AmberNeural"]}},
            "french": {
                "lang": "fr-FR",
                        "voices": {
                            "male": [
                                "fr-FR-HenriNeural",
                                "fr-FR-AlainNeural"],
                            "female": [
                                "fr-FR-DeniseNeural",
                                "fr-FR-JacquelineNeural"]}},
            "spanish": {
                "lang": "es-MX",
                "voices": {
                    "male": [
                        "es-MX-JorgeNeural",
                        "es-MX-LibertoNeural"],
                    "female": [
                        "es-MX-DaliaNeural",
                        "es-MX-LarissaNeural"]}},
            "japanese": {
                "lang": "ja-JP",
                "voices": {
                    "male": [
                        "ja-JP-KeitaNeural",
                        "ja-JP-DaichiNeural"],
                    "female": [
                        "ja-JP-NanamiNeural",
                        "ja-JP-AoiNeural"]}}}

        self.speakers = {"male": [], "female": []}

    def run_tts(self, message):
        sentence = message['translation']
        language = message['translation_language']
        voice = message['voice']
        sid = message['session_id']
        if language not in self.languages:
            raise Exception(
                f"Azure Speech doesn't currently support {language}. Currently configured languages: {', '.join(self.languages.keys())}")
        if sid not in self.speakers[voice]:
            self.speakers[voice].append(sid)

        voice = self.languages[language]['voices'][voice][self.speakers[voice].index(
            sid) % len(self.speakers[voice])]
        lang = self.languages[language]['lang']
        ssml = f"<speak version='1.0' xml:lang='{lang}' xmlns='http://www.w3.org/2001/10/synthesis' " \
            "xmlns:mstts='http://www.w3.org/2001/mstts'>" \
            f"<voice name='{voice}'>" \
            "<mstts:silence type='Sentenceboundary' value='20ms' />" \
            "<mstts:express-as style='lyrical' styledegree='2' role='SeniorFemale'>" \
            "<prosody rate='1.05'>" \
            f"{sentence}" \
            "</prosody></mstts:express-as></voice></speak> "
        result = self.speech_synthesizer.speak_ssml(ssml)
        print("⌨️ got azure tts result")
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            print("⌨️ returning result")
            yield result.audio_data[44:]
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(
                "Speech synthesis canceled: {}".format(
                    cancellation_details.reason))
            if cancellation_details.reason == CancellationReason.Error:
                print(
                    "Error details: {}".format(
                        cancellation_details.error_details))

    # generate a chat using Azure OpenAI based on the participant's most
    # recent speech
    def run_llm(self, messages, stream=True):
        print("generating chat")

        response = openai.ChatCompletion.create(
            api_type='azure',
            api_version='2023-06-01-preview',
            api_key=os.getenv("AZURE_CHATGPT_KEY"),
            api_base=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            deployment_id=os.getenv("AZURE_CHATGPT_DEPLOYMENT_ID"),
            stream=stream,
            messages=messages
        )
        return response
