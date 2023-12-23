import pathlib
from typing import cast
import openai
import entities
import pickle
import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

api_key = pathlib.Path("api.txt").read_text().strip("")
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = api_key

def generate_image(prompt: str):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']
    print(image_url)

generate_image("a white siamese cat")

# https://oaidalleapiprodscus.blob.core.windows.net/private/org-DHBMIfw4Dc1kTvMTbqxY7gIg/user-pqaE6LhyjNohZFdYZ1G0HG7t/img-BPgWWfMbiSnAhcIyQjiUCEQ8.png?st=2023-05-02T20%3A04%3A10Z&se=2023-05-02T22%3A04%3A10Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-05-02T19%3A52%3A13Z&ske=2023-05-03T19%3A52%3A13Z&sks=b&skv=2021-08-06&sig=JJ9i%2BY9s7oBehmxRb23m2OvOBoBTf6ZYHVjoEBvzU0w%3D