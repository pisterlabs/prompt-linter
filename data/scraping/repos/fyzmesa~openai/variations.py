import openai
import base64
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

openai.api_key = ""

i = 1

while i < 5:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    response = openai.Image.create_variation(
        image=open("outputs/20230308132244image.jpg", "rb"),
        model="image-alpha-001",
        size="1024x1024",
        response_format="b64_json"
        )

    image_b64 = response['data'][0]['b64_json']
    imgdata = base64.b64decode(image_b64)
    filename = 'outputs/%simage.jpg' %timestamp
    with open(filename, 'wb') as f:
        f.write(imgdata)
    i += 1
