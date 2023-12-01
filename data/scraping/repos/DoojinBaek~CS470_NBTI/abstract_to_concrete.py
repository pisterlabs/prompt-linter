import os
import openai

from dotenv import load_dotenv

def abstract_to_concrete(abstract_word):
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.Completion.create(
        model="davinci:ft-personal-2023-05-11-15-22-42",
        prompt = f"Q: What is the symbol often used to represent the word '{abstract_word}'? Give me one word.\nA:",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    # print(response)
    return response.choices[0].text.strip()
