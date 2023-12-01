import whisper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from chatbot import Chatbot
from utils.models_and_path import WHISPER_MODEL_NAME

class WhisperChatbot(Chatbot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.whisper_model = whisper.Whisper(WHISPER_MODEL_NAME)
        self._load_translation_engine()


    def response(self, audio):
        self._clean_audio()
        self._load_audio(audio)
        self._process_audio()

        en_result = super().response(self.text)
        result_translated = self._translate_text(text=en_result, source="en", target=self.lang)

        return self.transcribed_text, self.text, self.lang, en_result, result_translated


    def _load_translation_engine(self):
        self.translation_prompt = PromptTemplate(
            input_variables=["source", "target", "text"],
            template="Translate from language {source} to {target}: {text}?",
        )
        self.translation_chain = LLMChain(llm=self.LLM, prompt=self.translation_prompt)


    def _load_audio(self, audio):
        assert isinstance(audio, bytes), "Audio must be bytes"
        assert self.whisper_model, "Whisper model not loaded"

        # load audio and pad/trim it to fit 30 seconds
        self.audio = self.whisper_model.load_audio(
            self.whisper_model.pad_or_trim(
                audio
            ))


    def _process_audio(self):
        assert self.audio, "Audio not loaded"
        assert self.whisper_model, "Whisper model not loaded"

        # Make log-Mel spectrogram and move to the same device as the model
        mel = self.whisper_model.melspectrogram(self.audio).to(self.whisper_model.device)

        # Detcet language
        _, probas = self.whisper_model.detect_language(self.audio)
        self.lang = max(probas, key=probas.get)

        # Decode the audio
        options = self.whisper_model.DecodingOptions(fp16=False)
        self.transcribed_text = whisper.decode(self.whisper_model, mel, options).text

        # Check the language of the audio;
        # if it's english, use the transcribed text as is
        # else, translate it to english
        if self.lang == "en":
            self.text = self.transcribed_text
        else:
            # translate from detected lang to en
            self.text = self._translate_text(self.transcribed_text, self.lang, "en")


    def _translate_text(self, text, source, target):
        return self.translation_chain({
            "source": source,
            "target": target,
            "text": text
        })["result"]


    def _clean_audio(self):
        self.audio = None
        self.lang = None
        self.text = None
        self.transcribed_text = None
