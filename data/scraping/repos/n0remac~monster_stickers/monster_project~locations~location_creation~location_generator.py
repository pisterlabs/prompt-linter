import openai
import os
from tinydb import TinyDB, Query
from utils.fileio import download_image
from .location import LandscapeCard

def location_generator(amount: int = 1, path: str='/home/cameron/Projects/StickerMonsters/monster_project/'):
    '''Generates locations and saves it to tinyDB and posts to Django app.'''
    location_db = TinyDB('location_db.json')

    for i in range(amount):
        location = make_location(path)
        location_db.insert(location.__dict__())

    printed = Query()
    locations = location_db.search(printed.print_status == False)

    location_ids = []
    for location in locations:
        location_ids.append(location['uuid'])
    return location_ids

def make_location(path: str, num=''):
    '''Generates a location and returns it.'''
    
    landscape = LandscapeCard(path+'static/locationimages')
    image_path = f'{landscape.image_path}'.replace('.png', f'{num}.png')
    if os.path.exists(image_path):
        if num == '':
            num = 1
        else:
            num += 1
        return make_location(path, num)
    try:
        landscape.filename = f'{landscape.filename}'.replace('.png', f'{num}.png')
        landscape.image_path = image_path
        generate_location(image_path, landscape, 1)
    except Exception as e:
        print("Error generating location, trying again.")
        print(e)
        return make_location(path)
    
    landscape.post_landscape()
    qr_path = f"{path}locationqrcodes/{landscape.filename}"
    generate_qr_code(landscape.url, qr_path)
    landscape.qr_code_path = qr_path

    return landscape

def generate_location(path, landscape, amount):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Image.create(
    prompt=landscape.prompt,
    n=amount,
    size="512x512"
    )
    for image in response['data']:
        url = image['url']
        download_image(url, path)

def generate_qr_code(url, path):
    api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={url}"
    download_image(api_url, path)
