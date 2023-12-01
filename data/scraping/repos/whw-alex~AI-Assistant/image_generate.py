import os
import openai

openai.api_key = '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'
openai.api_base = "http://166.111.80.169:8080/v1"

def image_generate(content: str):

    response = openai.Image.create(
        prompt=content,
        size="256x256"
    )
    img = response['data'][0]['url']
    return img

if __name__ == "__main__":
    image_generate('A cute baby sea otter')