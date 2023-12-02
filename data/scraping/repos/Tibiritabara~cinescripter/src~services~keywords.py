import re
from typing import List

import openai

from utils.common import SettingsLoader
from utils.logs import logger


class Keyword:

    APP_NAME = "OPENAI_KEYWORDS"

    def __init__(self, **kwargs):
        self.options = SettingsLoader.load(
            self.APP_NAME,
            kwargs
        )

    def generate(self, text: str) -> List[str]:
        logger.debug("Generating keywords for text: %s", text)
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

        result = response.choices[0].text.lower().replace('text:', '').replace('keywords:', '').replace('\n', '')
        result = re.sub(r"[\d\.]", '', result)
        results = result.split(',')
        keywords = [keyword.replace('-', '').lstrip().rstrip() for keyword in results if keyword != '']
        for keyword in keywords:
            if len(keyword.split(" ")) > 3:
                keywords.remove(keyword)
        logger.debug("Generated keyword: %s", keywords)
        return keywords[:self.options.get("max_keywords")]
