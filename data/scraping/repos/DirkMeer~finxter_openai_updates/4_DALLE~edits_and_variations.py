from decouple import config
from pathlib import Path
from openai import OpenAI
from utils import image_downloader


client = OpenAI(api_key=config("OPENAI_API_KEY"))
current_directory = Path(__file__).parent


def dalle_editor(image_path: str, masked_image_path: str, prompt: str):
    response = client.images.edit(
        model="dall-e-2",
        image=open(image_path, "rb"),
        mask=open(masked_image_path, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    image_url = response.data[0].url
    image_downloader(image_url)
    return image_url


# dalle_editor(
#     image_path=f"{current_directory}/edit-original.png",
#     masked_image_path=f"{current_directory}/edit-masked.png",
#     prompt="Picture of a sanddune in the middle of the desert with a scary alien monster standing at the top of the sanddune. Desert, with a clear blue sky. Fierce strong muscular monster alien.",
# )


def dalle_variation(img_path: str, number_of_variations: int):
    response = client.images.create_variation(
        model="dall-e-2",
        image=open(img_path, "rb"),
        n=number_of_variations,
        size="1024x1024",
    )
    image_url = response.data[0].url
    image_downloader(image_url)
    return image_url


dalle_variation(
    img_path=f"{current_directory}/edit-original.png",
    number_of_variations=1,
)
