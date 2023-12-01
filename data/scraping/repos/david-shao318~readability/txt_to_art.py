import openai
import os

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


def gpt3(prompt, engine='gpt-3.5-turbo'):
    print(f'Passage: {prompt}')
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are an artistic and imaginative assistant."},
            {"role": "user", "content": "Describe an impressionist painting based on the following scene from a "
                                        "passage in literature:\n" + prompt},
        ]
    )
    result = response['choices'][0]['message']['content']
    print(f'Painting Description: {result}')
    return result


def dalle2(prompt):
    try:
        response = openai.Image.create(
            prompt=gpt3(prompt),
            n=1,
            size="512x512"
        )
    except openai.error.RateLimitError:
        print('Error: API Rate Limit Reached')
        return ""
    return response["data"][0]["url"]
