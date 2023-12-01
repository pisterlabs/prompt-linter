import openai
import os

class GPTPromptGeneratorSDK:

    api_key = str()
    question = str()

    def __init__(self, question):
        self.question = question

    def generate_qa(self):
        openai.api_key = os.getenv('OPENAI_API_KEY') # Store the OpenAI API key as an environmental variabl
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the most suitable engine
            prompt=f"{self.question}\n\nGenerate questions:",
            temperature=0.5,
            max_tokens=256
        )

        return response.choices[0].text.strip()
