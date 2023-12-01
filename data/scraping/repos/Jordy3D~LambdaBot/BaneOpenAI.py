import os
import requests

import openai

import secrets
openai.api_key = secrets.openaikey


####################
# OpenAI API calls #
####################

# no API access for account information yet, here's the URL instead 
# https://platform.openai.com/account/usage

class Model:
    # contains created, id, object, owner, permissions, and status 
    def __init__(self, created, id, object, owner, permissions, status):
        self.created = created
        self.id = id
        self.object = object
        self.owner = owner
        self.permissions = permissions
        self.status = status

def get_models():
    models = []
    response = openai.Engine.list()
    for model in response["data"]:
        new_model = Model(model["created"], model["id"], model["object"], model["owner"], model["permissions"], model["ready"])
        models.append(new_model)

    return models


class AIText:
    # contains the text, finish reason, created, prompt, id, model, object, and token counts
    def __init__(self):
        self.text = text = ""
        self.finish_reason = finish_reason = ""
        self.created = created = ""
        self.prompt = prompt = ""
        self.id = id = ""
        self.model = model = ""
        self.object = object = ""
        self.completion_tokens = ""
        self.prompt_tokens = ""
        self.total_tokens = ""

    def generate_text(self, prompt, model="gpt-3.5-turbo",max_tokens=100, temperature=0.5, top_p=1, frequency_penalty=0, presence_penalty=0):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        self.text = response["choices"][0]["text"]
        self.finish_reason = response["choices"][0]["finish_reason"]
        self.created = response["created"]
        self.prompt = prompt
        self.id = response["id"]
        self.model = response["model"]
        self.object = response["object"]
        self.completion_tokens = response["usage"]["completion_tokens"]
        self.prompt_tokens = response["usage"]["prompt_tokens"]
        self.total_tokens = response["usage"]["total_tokens"]

class AIChat:
    def __init__(self):
        self.text = text = ""

def generate_chat(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
    )
        
    return response["choices"][0]["message"]["content"]


# dictionary to define the size of the image
sizes = {"small": "256x256", "medium": "512x512", "large": "1024x1024"}

class AIImage:
    # contains urls to the images and the created time
    def __init__(self):
        self.urls = []
        self.prompt = ""
        self.created = ""
        self.image_path = ""

    def generate_image(self, prompt, num_images=1, size="medium", api_key=secrets.openaikey):
        response = openai.Image.create(
            prompt=prompt, 
            n=num_images, 
            size=sizes[size], 
            api_key=api_key,
        )
              
        self.created = response["created"]
        self.prompt = prompt
        self.urls = [image["url"] for image in response["data"]]

    def save_images(self):
        short_prompt = self.prompt.split(" ")[:4]
        short_prompt = "".join([char for char in short_prompt if char.isalnum()])

        if not os.path.exists(f"images/{short_prompt}"):
            os.makedirs(f"images/{short_prompt}")

        for i, url in enumerate(self.urls):
            response = requests.get(url)
            # save the image
            with open(f"images/{short_prompt}/{self.created}_{i}.png", "wb") as f:
                f.write(response.content)
                self.image_path = f"images/{short_prompt}/{self.created}_{i}.png"

            # save the metadata of the generation as a text file
            with open(f"images/{short_prompt}/{self.created}_{i}.txt", "w") as f:
                f.write(f"Prompt: {self.prompt}\n"\
                        f"Created: {self.created}\n"\
                        f"URL: {url}")


###################
# Other functions #
###################

show_words = [
    "give",
    "give me",
    "show",
    "show me",
    "display",
    "bring up",
]

picture_words = [
    "picture",
    "image",
    "photo",
    "pic",
    "drawing",
    "painting",
]

def ask_for_image(message):
    message = message.lower()

    if any(word in message[:len(message)//4] for word in show_words) and any(word in message for word in picture_words):
        return True
    else:
        return False




if __name__ == "__main__":
    prompt = "a crab with lightning powers"
    
    # ai_image = AIImage()
    # try:
    #     ai_image.generate_image(prompt)
    # except:
    #     print("Image generation failed")
        
    # ai_image.download_images()

    # models = get_models()
    # for model in models:
    #     print(model.id)

    # ai_text = AIText()
    # ai_text.generate_text(prompt)
    # print(f"Prompt:   \n{ai_text.prompt}")
    # print(f"Response: \n{ai_text.text}")


    # prompt = "Hello, how are you?"
    # response = generate_chat(prompt)
    # print(response)

    print(ask_for_image("what is a phrase that would show me a picture?"))
    print(ask_for_image("Show me a picture of a dog"))
    print(ask_for_image("What is a show about pictures of dogs?"))
    print(ask_for_image("give me a picture of a rabbit"))