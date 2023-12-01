
import openai
import requests
from PIL import Image
from instabot import Bot

try:
    import shutil
    shutil.rmtree('config')
except:
    pass

bot = Bot()
passwd = '*********'
bot.login(username = 'vix.bot', password = passwd)

def make_square(im, min_size=256, fill_color=(255,255,255,0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im = new_im.convert("RGB")
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

API_Key = 'sk-********************************'
openai.api_key = API_Key

image_resp = openai.Image.create(prompt="minecraft mobs, oil painting", n=1, size="512x512")
url = image_resp['data'][0]['url']

test_image = Image.open(requests.get(url, stream=True).raw)
new_image = make_square(test_image)

try:
    import os
    os.mkdir('to_upload')
except:
    pass

path = f'to_upload/automate.jpg'
new_image.save(path)

cap = '🔥 This image is created by OpenAI API and uploaded using InstaBot package written in python language 💡' 
bot.upload_photo(path, caption = cap)

try:
    import shutil
    shutil.rmtree('config')
except:
    pass
