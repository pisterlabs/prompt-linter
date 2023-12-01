import os
import openai
from typing import List
from dataclasses import dataclass

openai.api_type = os.getenv("API_TYPE")
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("API_KEY")


@dataclass
class Context:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer


class OpenAI:
    def __init__(self):
        self.engine = "text-davinci-002"
        self.temperature = 0.7
        self.max_tokens = 1024
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = None
        self.context_window = 5

        self._opener = "The following is a conversation with an AI assistant. The assistant is insightful, creative, friendly, and a little playful."
        self._context: List[Context] = [
            Context("Hello, who are you?",
                    "Hi there! I am a GitHub AI assistant here to answer your questions."),
            Context("What is GitHub?", "GitHub is a website and cloud-based service that helps developers store and manage their code, as well as track and control changes to their code")
        ]

    def _generate_prompt(self, prompt: str, context: List[Context] = []):
        self._context += context
        p = f"{self._opener}\n"
        for c in self._context[-self.context_window:]:
            p += f"\nHuman: {c.question}\nAI: {c.answer}\n"

        p += f"\nHuman: {prompt}\nAI: "
        return p

    def chat(self, prompt, context: List[Context] = []):
        p = self._generate_prompt(prompt, context)
        response = openai.Completion.create(
            engine=self.engine,
            prompt=p,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop
        )
        return {
            "response" : response.choices[0].text.strip(),
        }
    
    def set_context_window(self, window):
        self.context_window = window

    def set_engine(self, engine):
        self.engine = engine

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_top_p(self, top_p):
        self.top_p = top_p

    def set_frequency_penalty(self, frequency_penalty):
        self.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty):
        self.presence_penalty = presence_penalty

    def set_stop(self, stop):
        self.stop = stop


if __name__ == "__main__":
    oai = OpenAI()
    q = "What do you like about it the most?"
    r = oai.chat(q, [Context("Why are cars so noisy?", "Cars are noisy because they have engines."), 
                Context("What is the best way make them quieter?", "The best way to make them quieter is to use an electric car."),
                     Context("What about mileage?", "Electric cars have a longer mileage than gas cars.")])
    print(r)
