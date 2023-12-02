import os
import openai
import dotenv
import sys 
import argparse

parser = argparse.ArgumentParser(description='Create variation of image.')
parser.add_argument(
    'file',
    metavar='FILE',
    type=str,
    help='file to edit'
)
parser.add_argument(
    '-n',
    "--number",
    metavar='NUMBER',
    required=False,
    type=int,
    default=1,
    help='number of images to generate, default is 1'
)
parser.add_argument(
    '-s',
    "--size",
    metavar='SIZE',
    required=False,
    type=str,
    default="256x256",
    help='size of images to generate, default is 256x256'
)

args = parser.parse_args()

# Load the environment variables
dotenv.load_dotenv()
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except KeyError:
    print("OPENAI_API_KEY environment variable not found, exitting.")
    sys.exit(1)


# generate image from text prompt

response = openai.Image.create_variation(
    image=open(args.file, "rb"),
    n=args.number,
    size=args.size,
    response_format="b64_json"
)
# image_urls = response['data']  # [0]['url']

#save image to file
# import requests
from PIL import Image
from io import BytesIO
import base64
"""
for i in range(len(image_urls)):
    response = requests.get(image_urls[i]["url"])
    img = Image.open(BytesIO(response.content))
    img.save(f"images/img_{i}.png")
"""
data = response["data"]
for i in range(len(data)):
    img = Image.open(BytesIO(base64.b64decode(data[i]["b64_json"])))
    img.save(f"images/edit_{i}.png")
