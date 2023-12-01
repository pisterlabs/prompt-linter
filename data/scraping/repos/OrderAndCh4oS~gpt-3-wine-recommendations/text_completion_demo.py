import os
from pprint import pprint

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPEN_AI_API_KEY')

def main():
    prompt = "A complex, structured young chardonnay"
    completion = openai.Completion.create(
        engine="davinci:ft-orderandchaos-2022-07-24-14-52-09",
        prompt=prompt,
        temperature=0.9,
        top_p=1,
        frequency_penalty=1.75,
        presence_penalty=0,
        max_tokens=64,
        n=1
    )

    pprint(completion)

    print(prompt + ' ' + completion.choices[0].text)


if __name__ == '__main__':
    main()
