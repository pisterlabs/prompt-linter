import requests
import openai
openai.api_key = 'openai_api_key'
 
response = openai.Image.create(
  prompt='台灣地區九份的風景水彩',
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)
