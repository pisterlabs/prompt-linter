import openai
import os
import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

openai.api_key = os.environ.get('OPENAI_API_KEY')


def save_png(uuid: str):
    chat = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0.5)
    print("reading json model")
    with open(f"{uuid}.json") as json_file:
        data = json.load(json_file)
        json_content = json.dumps(data)

    print("fetching prompt")
    with open("prompts/plantuml_generator.txt") as plantuml_prompt_file:
        plantuml_prompt = plantuml_prompt_file.read()

    messages = [
        SystemMessage(
            content=plantuml_prompt
        ),
        HumanMessage(
            content=json_content
        ),
    ]

    print("writing plantuml diagram")
    with open(f"plantuml_diagrams/{uuid}.pu", "w+") as output_file:
        output_file.write(chat(messages).content)

    print("converting plantuml to png")
    os.system(f"java -jar plantuml.jar plantuml_diagrams/{uuid}.pu")
    os.system(f"mv plantuml_diagrams/{uuid}.png server/diagrams/{uuid}.png")
