import os
import random
from base64 import b64decode
import openai
from helperGR import upload_image_to_getresponse

OPEN_AI = os.environ['OPEN_AI']
openai.api_key = OPEN_AI
IMAGE_SAVE_FOLDER = "img"
STYLE_LIST = [
    "Impressionism Monet",
    "Cubism Picasso",
    "Street art graffiti",
    "Isometric 3D",
    "Low poly",
    "Digital painting",
    "Memphis ,bold, kitch, colourful, shapes",
    "realistic picture",
]

def get_image(subject):
    random_style = random.choice(STYLE_LIST)
    
    try:
        response = openai.Image.create(
            prompt=f"{random_style} of {subject}",
            n=1,
            size="256x256",
            response_format="b64_json",
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

    image_name = f"{response['created']}.png"
    image_data = response["data"][0]["b64_json"]
    img_url = save_image(image_data, image_name)

    return img_url
    
def save_image(image_data, image_name):
    decoded_image_data = b64decode(image_data)
    image_path = os.path.join(IMAGE_SAVE_FOLDER, image_name)
    with open(image_path, mode="wb") as png:
        png.write(decoded_image_data)
    img_url = upload_image_to_getresponse(image_path, image_name)
    return img_url    
        
        
liveurl = get_image("poppies")
print(liveurl)