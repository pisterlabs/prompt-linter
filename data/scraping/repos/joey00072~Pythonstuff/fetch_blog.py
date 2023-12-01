import requests
import json
import html2text
from datetime import datetime
from dotenv import load_dotenv
import os

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

BLOG_DIR = "./data/blog"

# Load Environment Variables
load_dotenv()
GHOST_URL = os.getenv("GHOST_URL")
GHOST_API = os.getenv("GHOST_API")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class ContentAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
        }

    def get_posts(self):
        endpoint = "/ghost/api/v3/content/posts/"
        params = (("key", self.api_key),)
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        posts = json.loads(response.text)

        return posts["posts"]

    @staticmethod
    def convert_html_to_markdown(html):
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        return converter.handle(html)

    def transform_post(self, post):
        markdown_post = {}
        markdown_post["title"] = post.get("title", "")
        markdown_post["date"] = (
            "'"
            + str(
                datetime.strptime(
                    post.get("created_at", ""), "%Y-%m-%dT%H:%M:%S.%f%z"
                ).strftime("%Y-%m-%d")
            )
            + "'"
        )
        markdown_post[
            "tags"
        ] = (
            []
        )  # you might need to update this part based on how you store tags in Ghost
        from pprint import pprint

        pprint(post)
        markdown_post["images"] = post.get("feature_image", "")
        markdown_post["content"] = self.convert_html_to_markdown(post.get("html", ""))
        markdown_post["summary"] = post.get("meta_description")
        markdown_post["tags"] = post.get("tags")
        content = f'---\ntitle: {markdown_post["title"]}\ndate: {markdown_post["date"]}\ntags: {markdown_post["tags"]}\n'

        if markdown_post["images"]:  # if image is present
            content += f'images: {markdown_post["images"]}\nsummary: {markdown_post["summary"]}\n---\n\n![]({markdown_post["images"]})\n\n'
        else:  # if no image is present
            content += f'summary: {markdown_post["summary"]}\n---\n\n'

        content += markdown_post["content"]
        return content

    @staticmethod
    def write_to_file(content, title):
        filename = os.path.join(BLOG_DIR, title.lower().replace(" ", "_") + ".md")
        with open(filename, "w") as file:
            file.write(content)


if __name__ == "__main__":
    api = ContentAPI(GHOST_URL, GHOST_API)
    posts = api.get_posts()

    for post in posts:
        markdown_post = api.transform_post(post)
        api.write_to_file(markdown_post, post["title"])
