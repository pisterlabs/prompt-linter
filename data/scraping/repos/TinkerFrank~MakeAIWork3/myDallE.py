#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "AZURE_DALLE_ENDPOINT"
openai.api_version = "2023-06-01-preview"
openai.api_key = os.getenv("AZURE_DALLE_KEY")

## libraries needed to show the image ##
from PIL import Image
import requests
from io import BytesIO

while (True):
    
    print('Input:')
    user_input = input("")

    if (user_input == 'q'):
        break

    response = openai.Image.create(
        prompt= user_input,
        size='1024x1024',
        n=1
    )

    image_url = response["data"][0]["url"]
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.show()

    from PIL import Image



