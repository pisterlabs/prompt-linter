import ssl

from openai import OpenAI
import urllib.request
import certifi
from github import Github
import requests
import os
from flask import Flask, request


def imgGenerator(textprompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.images.generate(
        model="dall-e-3",
        prompt=textprompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    return image_url


def urlDownloader(url):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ssl_context) as r, open("house.png", 'wb') as out_file:
        data = r.read()  # Read the data from the URL
        out_file.write(data)


app = Flask(__name__)
g = Github(os.environ["GITKEY"])


@app.route('/add', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        data = request.json
        imgurl = imgGenerator(data['prompt'])
        response = requests.get(imgurl)
        file_path = "House.png"
        repo = g.get_repo("prajyaguru003/AIATL-2024")
        try:
            contents = repo.get_contents(file_path, ref="main")
            repo.delete_file(contents.path, "Delete image", contents.sha, branch="main")
        except Exception:
            print("none exist")
        repo.create_file(file_path, "Add image", response.content, branch="main")


if __name__ == '__main__':
    app.run(debug=True)
