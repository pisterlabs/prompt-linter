import openai
import os

def chatbot_completion(message):

    api_key = os.environ["OPENAI_API_KEY"]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )

    return completion.choices[0].message.content
