"""
This module will perform retrieval augmented generation (RAG) over an image via access to
vision transformers hosted in vison_server.
"""

import logging
import os
import time
import requests
import json

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# import example store
try:
    from modules.image_rag_example_store import DittoImageRAGExampleStore
except:
    from image_rag_example_store import DittoImageRAGExampleStore

import base64
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("image_rag")
log.setLevel(logging.INFO)

TEMPLATE = """
The following is a user's query over an image coming from a smart home assistant's eyes or camera.
The smart assistant's name is Ditto, and is the user's AI friend and assistant.

Given the caption and the user's query, you have access to the following tools if needed to fully answer the user's query:

Tools:
1. Question Answering (QA) over the image
1.1 Command: <QA> query
1.2 Example: <QA> What is the color of the car?
2. <DONE> finish answering the user's query
2.1 Command: <DONE> response
2.2 Example: <DONE> I see a red car that is parked in the driveway. The car looks like a Tesla.

{examples}

Current Prompt:

user's query: {query}{caption}
response:
"""


class DittoImageRAG:
    def __init__(self):
        self.vision_server_ip = os.getenv("vision_server_ip")
        self.vision_server_port = int(os.getenv("vision_server_port", 52032))
        self.vision_server_protocol = os.getenv("vision_server_protocol", "http")
        self.vision_server_url = f"{self.vision_server_protocol}://{self.vision_server_ip}:{self.vision_server_port}"
        self.init_llm_agent()
        log.info("Initialized image rag agent")

    def check_if_vision_server_running(self):
        try:
            url = f"{self.vision_server_url}/status"
            requests.get(url)
            return True
        except BaseException as e:
            log.error(e)
            return False

    def init_llm_agent(self):
        self.llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo")
        self.prompt_template = PromptTemplate(
            input_variables=["examples", "query", "caption"],
            template=TEMPLATE,
        )
        self.example_store = DittoImageRAGExampleStore()

    def get_caption(self, image):
        if not self.check_if_vision_server_running():
            log.error("Vision server is not running")
            return None
        else:
            try:
                url = f"{self.vision_server_url}/caption"
                files = {"image": image}
                raw_response = requests.post(url, files=files)
                response = json.loads(raw_response.content.decode())["response"]
                return response
            except BaseException as e:
                log.error(e)
                return None

    def get_qa(self, query, image):
        if not self.check_if_vision_server_running():
            log.error("Vision server is not running")
            return None
        else:
            try:
                url = f"{self.vision_server_url}/qa"
                files = {"image": image}
                params = {"prompt": query}
                raw_response = requests.post(url, files=files, params=params)
                response = json.loads(raw_response.content.decode())["response"]
                return response
            except BaseException as e:
                log.error(e)
                return None

    def prompt(self, user_query, image, caption_image=False):
        log.info("Prompting image rag agent")
        print()
        print(f"user's query: {user_query}")

        if caption_image == False:
            caption = ""
        else:
            raw_caption = self.get_caption(image)
            caption = f"\nimage's caption: {raw_caption}"
            print(f"image's caption: {raw_caption}")

        # construct prompt with examples
        examples = self.example_store.get_examples(user_query)
        prompt = self.prompt_template.format(
            examples=examples, query=user_query, caption=caption
        )

        max_iterations = 5

        for i in range(max_iterations):
            res = self.llm.call_as_llm(prompt)
            if "<QA>" in res:
                llm_query = str(res).split("<QA>")[-1].strip().split("\n")[0]
                qa = self.get_qa(llm_query, image)
                llm_command = f"\n<QA> {llm_query}" + f"\n<QA Response> {qa}"
                prompt += llm_command
                print(llm_command)

            elif "<DONE>" in res:
                response = str(res).split("<DONE>")[-1].strip().split("\n")[0]
                print(f"\n<DONE> {response}")
                break

            if i == (max_iterations - 1):
                prompt += f"\n<DONE> "
                response = self.llm.call_as_llm(prompt)
                print(f"\n<DONE> {prompt+response}")
                break

        return response


if __name__ == "__main__":
    from PIL import Image

    image_rag = DittoImageRAG()
    image_path = "river-sunset.png"
    image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue())
    query = "Can you describe this image? I want to know where it is, what time of day it is, what the weather is like, and what color the sun is."
    # query = 'Tell me 2 things about this.'
    response = image_rag.prompt(user_query=query, image=base64_str, caption_image=True)
