import os
import openai



def get_completion(text, engine="text-davinci-002", temperature=0.5, max_tokens=60, top_p=1.0, frequency_penalty=0.8, presence_penalty=0.0):
    openai.api_key = "your_api_key"
    prompt = "Extract keywords from this text:\n\n" + text
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response







