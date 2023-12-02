from openai import OpenAI


def openai_response(model, messages):
    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)
    response_content = response.choices[0].message.content

    return response_content
