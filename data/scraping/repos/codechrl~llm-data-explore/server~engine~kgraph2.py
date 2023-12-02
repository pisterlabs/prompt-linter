# import streamlit as st
import json
from collections import Counter
from typing import List

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from setting import setting


class Node(BaseModel):
    subject: str = None
    relation: str = None
    subject_related: str = None


class KnowledgeGraph(BaseModel):
    node: List[Node]


model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
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
    Subject and subject is ONE to ONE relation.
    Answer only with the instuction below. No need explanation or anything not neccesary.
    
    {format_instructions}\n{text}\n
    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def generate_pydantic_iter(title, output_path="data/db"):
    print("graph")
    text_path = f"data/summary/{title}.txt"

    with open(text_path, "r") as file:
        text_data = file.read()

    text_data_split = text_data.split("\n")
    graph = []
    for idx, _text_data_split in enumerate(text_data_split):
        try:
            print(f"graph - {idx}")
            _input = prompt.format_prompt(text=_text_data_split)
            output = model(_input.to_string())
            extraction_output = parser.parse(output)
            graph.append(extraction_output)
        except Exception as exc:
            print(f"Error: {exc}")

    output = []
    for graph_elem in graph:
        output.extend(
            [
                {
                    "subject": node.subject,
                    "relation": node.relation,
                    "subject_related": node.subject_related,
                }
                for node in graph_elem.node
            ]
        )
    output_path = f"data/graph/{title}.json"
    with open(output_path, "w") as file:
        json.dump({"graph": output}, file)


def format(title):
    with open(f"data/graph/{title}.json", "r") as json_file:
        data = json.load(json_file)

    subjects = [elem["subject"] for elem in data["graph"]]
    objects = [elem["subject_related"] for elem in data["graph"]]

    subjects_counts = Counter(subjects)
    subjects_total_occurences = sum(subjects_counts.values())
    subjects_counts_list = [
        {"name": key, "occurrence": value} for key, value in subjects_counts.items()
    ]

    for idx, elem in enumerate(subjects_counts_list):
        subjects_counts_list[idx]["size_percentage"] = (
            elem["occurrence"] / subjects_total_occurences
        )

    for idx, elem_subj in enumerate(subjects_counts_list):
        objects = []
        for idx_graph, elem_graph in enumerate(data["graph"]):
            if elem_subj["name"] == elem_graph["subject"]:
                objects.append(elem_graph["subject_related"])
            subjects_counts_list[idx]["subject_related"] = objects

    return subjects_counts_list
