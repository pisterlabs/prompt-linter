import openai
from libs import constant

openai.api_key = constant.OPENAI_KEY


def get_summary(text, prompt="summarize this text: "):
    print("summry")
    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt + text, temperature=1, max_tokens=300
    )
    return response
