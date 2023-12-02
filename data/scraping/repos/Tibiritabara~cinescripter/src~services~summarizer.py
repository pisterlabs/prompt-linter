import openai

from utils.common import SettingsLoader
from utils.logs import logger


class Summarizer:

    APP_NAME = "OPENAI_SUMMARIZER"

    def __init__(self, **kwargs):
        self.options = SettingsLoader.load(
            self.APP_NAME,
            kwargs
        )

    def generate(self, text: str) -> str:
        logger.debug("Summarizing paragraph for text: %s", text)
        response = openai.Completion.create(
            model=self.options.get("model", "gpt-3.5-turbo"),
            prompt=self.options.get("prompt").substitute({
                "text": text,
            }),
            temperature=self.options.get("temperature"),
            max_tokens=self.options.get("max_tokens"),
            top_p=self.options.get("top_p"),
            frequency_penalty=self.options.get("frequency_penalty"),
            presence_penalty=self.options.get("presence_penalty"),
        )
        sentence = response.choices[0].text.replace('\n', '').rstrip().lower()
        logger.debug("Generated sentence: %s", sentence)
        return sentence
