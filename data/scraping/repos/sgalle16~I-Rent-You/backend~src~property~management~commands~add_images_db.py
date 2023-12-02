import time
from django.core.management.base import BaseCommand
from property.models import Property
import json
import os
import openai
import requests
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv('.env')
openai.api_key  = os.environ['openAI_api_key']

def fetch_image(url):
    response = requests.get(url)
    response.raise_for_status()

    # Convert the response content into a PIL Image
    image = Image.open(BytesIO(response.content))
    return(image)

class Command(BaseCommand):
    help = 'Add an image for each property title to the database'

    def handle(self, *args, **kwargs):
        items = Property.objects.all()
        for item in items:
            response = openai.Image.create(
                prompt=f"Alguna vista de la propiedad desde el exterior {item.title}",
                n=1,
                size="256x256"
            )
            image_url = response['data'][0]['url']
            img = fetch_image(image_url)
            img.save(f'media_root/media/property/property_images/{item.title}.jpg')           
            item.images = f'media/property/property_images/{item.title}.jpg'   
            item.save()

# The line `time.sleep(20)` is causing the program to pause execution for 20 seconds. This is done to
# ensure that there is a delay between each iteration of the loop. In this specific case, it is used
# to prevent making too many requests to the OpenAI API in a short period of time, which could
# potentially exceed rate limits or cause other issues.
            time.sleep(20)
            
        self.stdout.write(self.style.SUCCESS(f'Successfully updated item with ID {item_id}'))
        