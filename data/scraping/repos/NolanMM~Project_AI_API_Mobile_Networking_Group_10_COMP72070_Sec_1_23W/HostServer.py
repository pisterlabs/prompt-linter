import datetime
import configparser
from base64 import b64decode
import webbrowser
from fastapi import FastAPI
from typing import Optional
import openai
from openai.error import InvalidRequestError

app = FastAPI()

@app.get('/')
async def generate_image(Prompt: str = "Picture", num_image=1, size='512x512', output_format='url'):
    """
    params:
        prompt (str):
        num_image (int):
        size (str):
        output_format (str):
    """
    try:
        images = []
        response = openai.Image.create(
            prompt=Prompt,
            n=num_image,
            size=size,
            response_format=output_format
        )
        if output_format == 'url':
            for image in response['data']:
                images.append(image.url)
        elif output_format == 'b64_json':
            for image in response['data']:
                images.append(image.b64_json)
        return images
    except InvalidRequestError as e:
        print(e)


# config = configparser.ConfigParser()
# config.read('credential.ini')
# API_KEY = config['openai']['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']

openai.api_key = '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'

SIZES = ('1024x1024', '512x512', '256x256')

# generate images (url outputs)
# response = generate_image('cat with a book', num_image=2, size=SIZES[0])
# response['created']
# images = response['images']
# for image in images:
#    webbrowser.open(image)

# generate images (byte output)


@app.get('/Requests/{prompt}')
def requestAI(prompt: str):
    response = generate_image(prompt,
                              num_image=1, size=SIZES[1], output_format='b64_json')
    prefix = 'demo234'
    for indx, image in enumerate(response['images']):
        with open(f'{prefix}_{indx}.txt', 'wb') as f:
            f.write(b64decode(image))
