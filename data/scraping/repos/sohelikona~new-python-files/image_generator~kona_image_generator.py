from api_key import API_KEY

import openai


openai.api_key = API_KEY


response = openai.Image.create(
    prompt="very beautiful flower garder",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']

print(image_url)