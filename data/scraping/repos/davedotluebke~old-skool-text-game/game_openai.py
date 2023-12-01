import openai
import api_keys
openai.api_key = api_keys.OPENAI_API_KEY

def openai_completion(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.5
    )
    return response['choices'][0]['text']
