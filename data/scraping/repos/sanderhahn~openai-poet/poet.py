import os

import openai
from dotenv import load_dotenv

import personas

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class PoetException(Exception):
    pass

def generate_poem(
        persona_code: str,
        friend: str,
        occasion: str,
        memory: str,
        prompt_template: str
    ) -> str:

    selected_persona = personas.get_persona(persona_code)

    # Construct the prompt using the provided inputs
    prompt = prompt_template.format(
        friend=friend,
        occasion=occasion,
        memory=memory,
        persona_nickname=selected_persona["nickname"],
    )

    # Use the parameters in OpenAI API call
    try:
        nickname = selected_persona["nickname"]
        selected_persona["summary"]
        system_content = f"""
            You are a fictional poet called {nickname}.
            You write poems that are at least 10 lines long and you ensure each line
            ends with proper punctuation.
            Don't abruptly end poems and ensure end rhyme in the verse.
        """.strip()
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": system_content,
            }, {
                "role": "user",
                "content": prompt,
            }],
            temperature=selected_persona["temperature"],
            top_p=selected_persona["top_p"],
            frequency_penalty=selected_persona["frequency_penalty"],
            presence_penalty=selected_persona["presence_penalty"],
        )

        poem = chat_completion.choices[0].message.content.strip()
        return poem

    except openai.error.AuthenticationError as e:
        raise PoetException("OpenAI Auth Failed") from e
