import os
import openai
from dotenv import load_dotenv
load_dotenv()

class ml_backend:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_email(self, userPrompt="Write me a professionally sounding email", start="Dear"):
        """Returns a generated email using GPT3 with a certain prompt and starting sentence"""

        response = openai.Completion.create(
            engine="davinci",
            prompt=userPrompt + "\n\n" + start,
            temperature=0.71,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.36,
            presence_penalty=0.75
        )
        return response.choices[0]['text']

    def replace_spaces_with_pluses(self, sample):
        """Returns a string with each space being replaced with a plus so the email hyperlink can be formatted properly"""
        changed = list(sample)
        for i, c in enumerate(changed):
            if c == ' ' or c == '  ' or c == '   ' or c == '\n' or c == '\n\n':
                changed[i] = '+'
        return ''.join(changed)
