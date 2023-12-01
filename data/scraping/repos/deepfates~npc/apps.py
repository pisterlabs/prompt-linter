from openai import Image # type: ignore


import requests
import os
import shutil
from pathlib import Path


from asyncstdlib.functools import lru_cache # type: ignore
from dotenv import load_dotenv
load_dotenv()

def get_dalle_template(text):
    return f"{text} fantasy horror rpg, first person cinematic beautiful warm light #ffe466"

def download_image(url):
    """Takes a remote image URL and downloads it to the local drive in public/assets/gen. It returns the local URL."""

    # Get the filename from the URL (looks like https://oaidalleapiprodscus.blob.core.windows.net/private/org-NRNQIoEWpUsCmwbmlvL5A87L/user-Up0I6WIVpyHVPSbxTD0yjFz4/img-zWjxTreQRk0yoYxKGITSLRky.png?st=2023-02-07T21%3A21%3A45Z&se=2023-02-07T23%3A21%3A45Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-02-07T21%3A33%3A10Z&ske=2023-02-08T21%3A33%3A10Z&sks=b&skv=2021-08-06&sig=H0gnr2rc3MLm2QjsmrzoC5Im8BzzqEMlA2XPGEf6tJw%3D)
    # We only want the part like "img-zWjxTreQRk0yoYxKGITSLRky.png"
    filename = url.split("/")[6].split("?")[0]
    # Create the path to the local file
    local_path = Path(os.getcwd()) / "client" / "public" / "assets" / "gen" / filename

    # Download the file
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        print(f"Error downloading image {url}")

    # Return the local URL
    return f"/assets/gen/{filename}"

@lru_cache(maxsize=None)
async def generate_image(text):
    """Get an image from OpenAI's DALL-E model."""
    # print(f"Generating image for {text}")
    response = Image.create(
        prompt=get_dalle_template(text),
        n=1,
        size="256x256",
        )

    url = response['data'][0]['url']
    url = str(url)
    local_url = download_image(url)
    return local_url

def get_template(token):
    def template(text):
        return f"Adventure game screencap, beautiful graphics, {token}:{text} first person beautiful graphics, CRT screen vignette scanlines"
    return template


from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document

class Summarizer():

    def __init__(self) -> None:
        self.llm = OpenAI(temperature=0.0, max_tokens=60)
        self.chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        self.cache: dict[str, str] = {}

    def run(self, text):
        if text in self.cache:
            return self.cache[text]
        doc = Document(page_content=text)
        summary = self.chain.run([doc])
        self.cache[text] = summary
        return summary
        