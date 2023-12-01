import openai
import base64
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")

openai.api_key = ""

##############################################################################

prompt = "a realistic photography of a crocodile smiling and wearing cosmonauts outfits with a moon in the background"

##############################################################################

i = 1

while i < 2:
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    response = openai.Image.create_edit(
        image=open("outputs/20230303133324image.jpg", "rb"),
        mask=open("outputs/mask.png", "rb"),
        prompt=prompt,
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
