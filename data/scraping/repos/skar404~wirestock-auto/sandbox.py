import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
from os import path
from typing import Tuple

import pandas as pd

import openai
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from aio_clients import Http, Options
from aiohttp import ClientTimeout
from geopy import Location
from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter

# - описание на английском

ASTICA_API_KEY = ''
OPENAI_KEY = ''


@dataclass
class ImageDescription:
    coordinates: Tuple[float, float] = None
    camera: str = None
    created: str = None

    location_str: str = None
    location: Location = None
    description: object = None


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, value) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[value] = exif[idx][t]

    return geotagging


def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_coordinates(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat, lon)


async def get_img_decryption(img: str, url: str):
    """Получение описания изображения с помошью asctica"""

    http = Http(
        option=Options(
            timeout=ClientTimeout(total=30),
        )
    )

    img_data = ImageDescription()

    if img:
        base_path = '---'
        p_image = Image.open(path.join(base_path, img))
        exif = p_image._getexif()

        p_image.resize((1024, 1024))

        coordinates = None
        try:
            geotags = get_geotagging(exif)
            coordinates = get_coordinates(geotags)
        except Exception as ex:
            print('geo tagging error err=', ex)

        # p_img.save('test.jpg')
        buffered = BytesIO()
        p_image.save(buffered, format="jpeg")
        img_str = base64.b64encode(buffered.getvalue())

        img_data.coordinates = coordinates
        img_data.camera = exif[271]
        img_data.created = exif[36868]

        if coordinates:
            async with Nominatim(
                    user_agent="github.com/skar404/wirestock-auto/develop",
                    adapter_factory=AioHTTPAdapter,
                    timeout=10,
            ) as geolocator:
                img_data.location = await geolocator.reverse(coordinates, language='en')

    if not img_data.location:
        img_data.location_str = input('location\n>')

    r = await http.post(
        path="https://vision.astica.ai/describe",
        data={
            "tkn": ASTICA_API_KEY,
            "modelVersion": "2.1_full",
            "visionParams": "gpt",
            "input": f"data:image/jpeg;base64,{img_str.decode('utf-8')}" if img else url
        }
    )
    img_data.description = r.json

    data = "Твоия задача присылать рецензию на фотографии которое прислала пользовательл для сервиса wirestock, правила:\n" \
           "- Пищи твое только рецензию фотографии\n" \
           "- В рецензии напиши локацию где сделано фото\n" \
           "- При написании учитывай  дату и время когда сделано фото\n" \
           "- Лимит описание 300 символов\n" \
           "- На английском языке\n" \
           "- Учитывай данные пользователя\n" \
           "Принимай тольок эти поля: \n" \
           "- Локация\n" \
           "- Дату и время\n" \
           "- Тестовая информация о фото\n"
    openai.api_key = OPENAI_KEY

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'system',
                'content': data,
            },
            {
                'role': 'user',
                'content': f"Локация: {img_data.location.address if img_data.location else img_data.location_str}\n"
                           f"{'Дату и время: ' + img_data.created if img_data.created else ''}\n"
                           f"Тестовая информация о фото: {img_data.description['caption_GPTS']}"
            }
        ],
    )
    print(response['choices'][0]['message']['content'])


if __name__ == '__main__':
    openai.api_key = OPENAI_KEY

    img_name = ''

    url = ''
    asyncio.run(get_img_decryption(img=img_name, url=url))
