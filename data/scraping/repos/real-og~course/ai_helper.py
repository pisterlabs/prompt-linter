import os
import openai

openai.api_key = str(os.environ.get('openai_token'))

def get_explanation(bar):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Explain this line from the song in few words: {bar}",
        temperature=0.25,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response["choices"][0]["text"]


