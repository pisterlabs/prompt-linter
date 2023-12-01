# import streamlit as st
import json

import openai
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
)
from pydantic import BaseModel
from setting import setting

default_schema = {
    "properties": {
        "subject": {"type": "string"},
        "relation": {"type": "string"},
        "subjectRelated": {"type": "string"},
    },
    "required": ["subject", "relation", "subjectRelated"],
}

llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=setting.OPENAI_API_KEY
)

chain = create_extraction_chain(default_schema, llm, verbose=True)
# chain = create_extraction_chain_pydantic(pydantic_schema=DefaultSchema, llm=llm)


def generate(title, output_path="data/db"):
    text_path = f"data/summary/{title}.txt"

    with open(text_path, "r") as file:
        text_data = file.read()

    extraction_output = chain(text_data, include_run_info=True)
    # markdown_output = json_to_markdown_table(extraction_output["text"])
    # run_id = extraction_output["__run"].run_id

    output = extraction_output["text"]
    output_path = f"data/graph/{title}.json"
    with open(output_path, "w") as file:
        json.dump({"graph": output}, file)


class KnowledgeGraph(BaseModel):
    subject: str = None
    relation: str = None
    subject_related: str = None


model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo-16k"
temperature = 0.0
model = OpenAI(
    model_name=model_name,
    temperature=temperature,
    openai_api_key=setting.OPENAI_API_KEY,
    # max_tokens=2000,
)

parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
prompt = PromptTemplate(
    template="""You are expert in building Knowlede Graph. 
    Identify subjects and its relation. 
    Subject and subject related must a noun.
    
    {format_instructions}\n{text}\n
    
    Example answer without format:
        subject : ChatGPT
        relation : part
        subject_related: LLM
        """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def generate_pydantic(title, output_path="data/db"):
    text_path = f"data/summary/{title}.txt"

    with open(text_path, "r") as file:
        text_data = file.read()

    _input = prompt.format_prompt(text=text_data)
    output = model(_input.to_string())
    extraction_output = parser.parse(output)

    output = extraction_output
    output_path = f"data/graph/{title}.json"
    with open(output_path, "w") as file:
        json.dump({"graph": output}, file)


def generate_pydantic_iter(title, output_path="data/db"):
    print("graph")
    text_path = f"data/summary/{title}_brief.txt"

    with open(text_path, "r") as file:
        text_data = file.read()

    text_data_split = text_data.split("\n")
    graph = []
    for idx, _text_data_split in enumerate(text_data_split):
        print(f"graph - {idx}")
        _input = prompt.format_prompt(text=_text_data_split)
        output = model(_input.to_string())
        extraction_output = parser.parse(output)
        graph.append(extraction_output)

    output = extraction_output
    output_path = f"data/graph/{title}.json"
    with open(output_path, "w") as file:
        try:
            json.dump({"graph": output}, file)
        except:
            file.write(output)


def call_openai_api_graph(chunk):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "Hello I am KnowledgeGraphBot. What can I help you?",
            },
            {
                "role": "user",
                "content": f""" Please identify all subject of the following text.
                                Then identify the relations.
                                Answer with this format of list of dict:
                                    [{{
                                        "subject": "ChatGPT,
                                        "relation" "part",
                                        "subject_related": "LLM
                                    }},
                                    {{
                                        "subject": "Claude,
                                        "relation" "part",
                                        "subject_related": "LLM
                                    }}]
                                Aanser ONLY with the list of dict. no explanation whatsoever.

                                TEXT: {chunk}
                                ANSWER:
                                """,
            },
        ],
        # max_tokens=15000,
        n=1,
        stop=None,
        temperature=0.1,
    )

    return response.choices[0]["message"]["content"].strip()


def generate_call(title, output_path="data/db"):
    text_path = f"data/summary/{title}_brief.txt"

    with open(text_path, "r") as file:
        text_data = file.read()

    extraction_output = call_openai_api_graph(text_data)

    output = extraction_output
    try:
        output = json.loads(output)
    except:
        pass
    output_path = f"data/graph/{title}.json"
    with open(output_path, "w") as file:
        json.dump({"graph": output}, file)
