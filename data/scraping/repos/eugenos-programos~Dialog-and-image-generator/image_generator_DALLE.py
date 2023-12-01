import argparse 
import json
import openai
import shutil
import requests
import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


openai.api_key = os.environ.get("OPENAI_API_KEY")

parser = argparse.ArgumentParser(
    prog="text2image"
)

parser.add_argument("-f", "--file", help="JSON file path", required=True, type=str)
parser.add_argument("--img-size", help="generated image size (256 | 512 | 1024)", default=256, type=int)

args = parser.parse_args()
file_path = args.file
img_size = args.img_size


def generate(text: str, img_size: int) -> str:
  res = openai.Image.create(
    prompt=text,
    n=1,
    size=f"{img_size}x{img_size}",
  )
  return res["data"][0]["url"]


with open(file_path) as file:
    words = json.load(file)


gauth = GoogleAuth()        
gauth.LocalWebserverAuth()   
drive = GoogleDrive(gauth)  


for prof_level in words.keys():
    for word_dict in words[prof_level]:
        word = word_dict["word"]
        img_file_path = f"images/{word}.png"
        response = requests.get(generate(word, img_size), stream=True)
        if response.status_code == 200:
            with open(img_file_path, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
        else:
            print(f"Status code - {response.status_code}")

        gd_file = drive.CreateFile({'title': f"{word}.png"})
        gd_file.SetContentFile(img_file_path)
        gd_file.Upload()

        gd_file = None
        print(f"Image for {word} is saved")
        