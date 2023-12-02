import os
import openai


def images_edit(image, prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.Image.create_edit(image=image, prompt=prompt)
        return response["data"][0]["url"]
    except Exception as e:
        print(e)
        return "Error"
