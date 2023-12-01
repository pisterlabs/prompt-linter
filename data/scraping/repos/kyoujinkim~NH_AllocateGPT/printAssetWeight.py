from typing import List

import openai
import tiktoken

from src.template import weightTemplate
from src.template import recommendTemplate


class printAssetWeight:
    def __init__(self,
                 llmAiEngine: str = 'gpt-3.5-turbo',
                 ):
        self.Engine = llmAiEngine
        self.tokenizer = tiktoken.encoding_for_model(llmAiEngine)

    def printAssetWeight(self, assetView: dict, constraint: List[str] = [], method: str = 'weight'):

        if method == 'weight':
            template = weightTemplate.loadTemplate(assetView, constraint)
        elif method == 'recommend':
            template = recommendTemplate.loadTemplate(assetView, constraint)

        response = openai.ChatCompletion.create(
            model=self.Engine,
            messages=[
                {"role": "assistant", "content": template}
            ],
            temperature=1.0,
        )

        text = response['choices'][0]['message']['content']

        return text
