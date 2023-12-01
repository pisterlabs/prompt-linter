import openai
def ver_ger(api_key):

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are fun"},
            {"role": "user", "content": "Сгенерируй забавное название для версии ПО. Просто выведи его без своих пояснений"},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return f"codename: {result}"