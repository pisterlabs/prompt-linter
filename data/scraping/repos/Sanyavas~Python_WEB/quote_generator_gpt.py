import json
import os
from pathlib import Path
import environ
from datetime import datetime

import openai

from ..models import Quote, Tag, Author

BASE_DIR = Path(__file__).resolve().parent.parent
env = environ.Env()
current_dir = os.path.dirname(os.path.abspath(__file__))
gpt_resp_json = os.path.join(current_dir, 'json', 'gpt_resp.json')

environ.Env.read_env(BASE_DIR / '.env')
api_key = env('API_KEY_OPENAI')
openai.api_key = api_key


def generate_quote():
    """
     The function uses the OpenAI API to generate a quote about poetry.
    """
    try:
        worker = "You are a creative writer."
        prompt = """Create creative quote about poetry, poetry should be on love style.
        Quote should be on English language, and contain about 300 symbols
        Write only quote, don't add any recommendation and explanations"""
        print(f"Start request to GPT")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": worker},
                {"role": "user", "content": prompt}
            ]
        )
        with open(gpt_resp_json, "w") as file:
            json.dump(response, file, indent=4, ensure_ascii=False)
        total_tokens = response.get("usage").get("total_tokens")
        print("====================")
        print(f'Total tokens: {total_tokens}. Date: {datetime.now()}')
        print("====================")
        return response.choices[0].message.content.strip()
    except Exception as ex:
        return f'Exception generate GPT quote: {ex}'


def add_quote_to_db(quote: str):
    """
    The add_quote_to_db function takes a string as an argument and adds it to the database.
    """
    author = Author.objects.filter(fullname="ChatGPT").first()
    tag = Tag.objects.get(name="poetry")
    new_quote = Quote.objects.create(quote=quote, author=author)
    new_quote.tags.add(tag)
    new_quote.save()
    print(f"New quote was added")


def gpt_creator():
    """
    The gpt_creator function is a function that generates quotes using the GPT-3.5 model.
    """
    print(f"Start generation quote")
    quote = generate_quote()
    add_quote_to_db(quote)


if __name__ == "__main__":
    print(generate_quote())
