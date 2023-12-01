# 1. Jak użyć OpenAI API w Pythonie?
## "Stop"

import openai


if __name__ == '__main__':
    openai.api_key_path = 'api.key'

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Oto jest",
        max_tokens=40,
        temperature=1.2,
        n=5,
        stop="\n"
    )

    for choice in response.choices:
        print(choice.text)
        print('-' * 80)