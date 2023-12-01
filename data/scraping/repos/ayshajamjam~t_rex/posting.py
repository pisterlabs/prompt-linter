import os
import glob
# cookie_del = glob.glob("config/*cookie.json")
# os.remove(cookie_del[0])
if os.path.isfile("path/to/config/file.json"):
    os.remove("path/to/config/file.json")

from query import *
from openai_api import *

def main():
    configure()

    # Determine category ['bucket_list', 'history', 'hobbies'] API options
    category = get_category()

    # Determine topic to prompt openAI API with
    topic = get_topic(category)

    # Get prompt
    prompt = get_prompt(category, topic)

    print("Category: ", category)
    print("Prompt :", prompt)

    # Get caption
    caption = get_caption(prompt)

    # Get image
    img_url = get_image_url(topic)

    # Save image in photos
    save_img(img_url)

    print("Caption: ", caption)
    print("Url: ", img_url)

main()