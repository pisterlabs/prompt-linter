import whisper
import openai

from ..config.whisper_config import *

import logging

root_dir = Path(__file__).parent

# Initialize logger with the given name
logger = logging.getLogger(__name__)
logger.debug("Initialized")


class SpeechInference(object):
    def __init__(self):
        """
        Initialize the speech inference class.
        """

        self.offline_mode = offline_mode

        if self.offline_mode:
            logger.debug("Initializing Speech Inference model")
            self.model = whisper.load_model(model_size)
            logger.debug("Initialized Speech Inference model")
        else:
            logger.debug("Initializing OpenAI API")
            openai.api_key = api_key
            logger.debug("Initialized OpenAI API")

    def run_stt(self):
        """
        Run speech inference.
        :return:
        """
        if self.offline_mode:
            logger.debug("Started Inferencing using offline method")

            result = self.model.transcribe(audio_file)
        else:
            logger.debug("Started Inferencing using online method")

            file = open(audio_file, "rb")
            result = transcription = openai.Audio.transcribe("whisper-1", file)

        return result["text"]


SpeechInferencer = SpeechInference()
