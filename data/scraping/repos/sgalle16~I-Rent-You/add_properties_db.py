from django.core.management.base import BaseCommand
from property.models import Property, PropertyFeature, PropertyImage
import json
import time
import os
import openai
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv('../../.env')
openai.api_key = os.environ['openAI_api_key']


def fetch_image(url):
    response = requests.get(url)
    response.raise_for_status()

    # Convert the response content into a PIL Image
    image = Image.open(BytesIO(response.content))
    return (image)


class Command(BaseCommand):
    help = 'Generate images & Load properties data from properties_complete.json into the property model, features and images to the property model'

    def handle(self, *args, **kwargs):
        # Load properties from JSON file
        with open('/home/shaan/I-Rent-You/properties_complete.json', 'r') as file:
            properties_data = json.load(file)

        # Initialize a counter for added properties
        cont = 0

        # Loop through each property in the JSON data
        for property_data in properties_data:
            # Check if the property already exists based on the title
            existing_property = Property.objects.filter(
                title=property_data['title']).first()

            # This code block checks if a property with the same title already exists in the database.
            # If `existing_property` is `None`, it means that there is no property with the same title
            # in the database. In this case, a new property is created using the
            # `Property.objects.create()` method. The property's attributes are set using the
            # corresponding values from the `property_data` dictionary.
            if existing_property is None:
                # Property doesn't exist, create a new one
                new_property = Property.objects.create(
                    title=property_data['title'],
                    description=property_data['description'],
                    #images=f'media/property/property_images/{property_data["title"]}.jpg',
                    type_of_property=property_data['type_of_property'],
                    time_for_rent=property_data['time_for_rent'],
                    location=property_data['location'],
                    address=property_data['address'],
                    size=property_data['size'],
                    rental_price=property_data['rental_price'],
                    status=property_data['status']
                )
                # Create an instance of PropertyFeature associated with the new Property
                # This block of code is responsible for creating and saving a `PropertyFeature` object
                # associated with a new `Property` object.
                property_feature_data = property_data.get('property_feature')
                if property_feature_data:
                    property_feature = PropertyFeature(
                        property=new_property,
                        num_bedrooms=property_feature_data.get('num_bedrooms'),
                        num_bathrooms=property_feature_data.get(
                            'num_bathrooms'),
                        parking_spaces=property_feature_data.get(
                            'parking_spaces'),
                        garden=property_feature_data.get('garden'),
                        pool=property_feature_data.get('pool'),
                        backyard=property_feature_data.get('backyard'),
                        furnished=property_feature_data.get('furnished')
                    )
                    property_feature.save()

                # Generate and save multiple images
                for i in range(3):  # Change 3 to the number of images you want
                    response = openai.Image.create(
                        prompt=f"Alguna vista de la propiedad desde el exterior {property_data['title']}",
                        n=1,
                        size="256x256"
                    )
                    image_url = response['data'][0]['url']
                    img = fetch_image(image_url)
                    img.save(f'media_root/media/property/property_images/{property_data["title"]}_{i}.jpg')
                    property_image = PropertyImage(
                        property=new_property,
                        images=f'media/property/property_images/{property_data["title"]}_{i}.jpg',
                        is_main_image=(i == 0)  # Set the first image as the main image
                    )
                    property_image.save()
                time.sleep(25)
                cont += 1
            else:
                # Property already exists, you can choose to update its information if needed
                pass

        self.stdout.write(self.style.SUCCESS(
            f'Successfully added {cont} properties, images and features to the database'))
