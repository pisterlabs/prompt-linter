import ast
from enum import Enum

import openai


import logging
from typing import List

import io
import base64
from PIL import Image
from aiohttp import ClientResponseError
from openai.openai_response import OpenAIResponse

from bobweb.bob import openai_api_utils, async_http
from bobweb.bob.openai_api_utils import ResponseGenerationException
from bobweb.bob.utils_common import split_to_chunks

logger = logging.getLogger(__name__)

# Dallemini api base url hosted by Craiyon.com
DALLEMINI_API_BASE_URL = 'https://bf.dallemini.ai/generate'

# dict for getting Openai Dall-e api expected image size string
image_size_int_to_str = {256: '256x256', 512: '512x512', 1024: '1024x1024'}


class ImageGenerationResponse:
    def __init__(self, images: List[Image.Image], additional_description: str = None):
        self.images = images or []
        self.additional_description = additional_description or ''


class ImageGeneratingModel(Enum):
    """
        Supported image generating models:
        - DALLEMINI - dalleminimodel hosted by Craiyon.com
        - DALLE2 - OpenAI's Dall-e 2 model using OpenAi's API
    """
    DALLEMINI = 1
    DALLE2 = 2


async def generate_images(prompt: str, model: ImageGeneratingModel) -> ImageGenerationResponse:
    """
    Generates image with given prompt and model. May raise exception.
    :param prompt: prompt passed to image generating model
    :param model: model to be used
    :return: List of images
    """
    match model:
        case ImageGeneratingModel.DALLEMINI:
            return await generate_dallemini(prompt)
        case ImageGeneratingModel.DALLE2:
            return await generate_using_openai_api(prompt)


async def generate_dallemini(prompt: str) -> ImageGenerationResponse:
    request_body = {'prompt': prompt}
    headers = {
        'Host': 'bf.dallemini.ai',
        'Origin': 'https://hf.space',
    }
    try:
        content: bytes = await async_http.post_expect_bytes(DALLEMINI_API_BASE_URL, json=request_body, headers=headers)
        images = get_images_from_response(content)
        image_compilation = get_3x3_image_compilation(images)
        return ImageGenerationResponse([image_compilation])
    except ClientResponseError as e:
        logger.error(f'DalleMini post-request returned with status code: {e.status}')
        raise ResponseGenerationException('Kuvan luominen epäonnistui. Lisätietoa Bobin lokeissa.')


async def generate_using_openai_api(prompt: str, image_count: int = 1, image_size: int = 1024) -> ImageGenerationResponse:
    """
    API documentation: https://platform.openai.com/docs/api-reference/images/create
    :param prompt: prompt used for image generation
    :param image_count: int - number of images to generate
    :param image_size: int - image resolution (height and width) that is used for generated images
    :return: List of Image objects
    """
    openai_api_utils.ensure_openai_api_key_set()

    response: OpenAIResponse = await openai.Image.acreate(
        prompt=prompt,
        n=image_count,
        size=image_size_int_to_str.get(image_size),  # 256x256, 512x512, or 1024x1024
        response_format='b64_json',  # url or b64_json
    )

    images = []
    for open_ai_object in response.data:
        base64_str = open_ai_object['b64_json']
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

        image.thumbnail((image_size, image_size))
        images.append(image)

    additional_description = openai_api_utils.state.add_image_cost_get_cost_str(image_count, image_size)

    return ImageGenerationResponse(images, additional_description)


def get_images_from_response(content: bytes) -> List[Image.Image]:
    json_content = ast.literal_eval(content.decode('UTF-8'))
    return convert_base64_strings_to_images(json_content['images'])


def get_3x3_image_compilation(images):
    # Assumption: All images are same size
    i_width = images[0].width if images else 0
    i_height = images[0].height if images else 0

    canvas = Image.new('RGB', (i_width * 3, i_height * 3))

    image_rows = split_to_chunks(images, 3)
    for (r_index, r) in enumerate(image_rows):
        for (i_index, i) in enumerate(r):
            x = i_index * i_width
            y = r_index * i_height
            canvas.paste(i, (x, y))
    return canvas


def convert_base64_strings_to_images(base_64_strings) -> List[Image.Image]:
    images = []
    for base64_str in base_64_strings:
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        images.append(image)
    return images
