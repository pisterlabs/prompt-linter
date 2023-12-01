import os
import re
import json
import openai

from .data_access import DataAccess

class SummaryAgent:
    def __init__(self) -> None:
        self.key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.key

        self.model = 'gpt-3.5-turbo-0613'

    def extract_subjects(self, text: str) -> list[str]:

        self.messages = [{"role": "user", 
                          "content": "Summarize subjects in the following message. " +
                                     "Strictly only output itemized subjects, no extra words.\n" + text}]
        
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )

        result = completion.choices[0]["message"]["content"]
        result = result.split('\n')

        return self.remove_numbering(result)
    
    @staticmethod
    def remove_numbering(items):
        return [re.sub(r'^[\d\-\s]*\.\s*', '', item) for item in items]