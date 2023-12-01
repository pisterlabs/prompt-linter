import json
import openai
from time import time as timer

from ui import info, warn, ok, end_color
from consts import MODEL, languages


def translate(data: dict | str, language_from: str, language_to: str, debug=False) -> dict:
    if type(data) is dict:
        data = json.dumps(data, indent=2)

    language_from = languages[language_from]
    language_to = languages[language_to]

    print(f'{info} Language from: {language_from}')
    print(f'{info} Language to: {language_to}')

    # if debug:
    #     print(f'{info} Text to translate:', data)

    prompt = f"""
    Translate the following JSON file from {language_from} to {language_to},
    following the same structure and following the JSON rules.

    {data}

    Return only JSON content.
    """

    if debug:
        print(f'{info} Calling OpenAI API...')

    start = timer()
    response = openai.ChatCompletion.create(
        engine='hv-gpt-35-turbo',
        model=MODEL,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    end = timer()

    if debug:
        print(f'{info} Request took {warn}{end - start:.2f} sec{end_color}')

    # if debug:
    #     print(response)

    return json.loads(response.choices[0].message.content)
