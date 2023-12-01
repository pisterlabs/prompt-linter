import openai

def tldr( text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{text}\n\nTl;dr",
        temperature=0,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]["text"]
        