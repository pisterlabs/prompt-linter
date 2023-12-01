import openai
import numpy as np
openai.api_key = "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"

class Blackboard:
    def __init__(self):
        self.words = []
    def write(self):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Correct this to standard English: {}".format(''.join(self.words)+'.'),
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        text = response["choices"][0]["text"]
        print(text)
    def add_word(self, word):
        if(self.words[-1]!=word):
            self.words.append(word)
    def clear(self):
        self.words = []
    def backspace(self):
        self.words.pop()

