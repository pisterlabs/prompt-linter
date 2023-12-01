
import os
import openai 


def gpt3(text):
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Completion.create(
        engine='davinci-instruct-beta',
        prompt=text,
            temperature=0.5,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
    )
    content = response.choices[0].text.split('.')
    return response.choices[0].text 

query = 'Write a short story about two dogs.'
response = gpt3(query)
print(response)