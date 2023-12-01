# 1. Jak użyć OpenAI API w Pythonie?

import openai


if __name__ == '__main__':
    openai.api_key_path = 'api.key'

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Litwo, Ojczyzno moja!",
        max_tokens=15,
        temperature=0
    )
    print(response.choices[0].text)