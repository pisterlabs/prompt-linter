import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from constant import GPT_4, GPT_3_5_TURBO, GPT_3_5_TURBO_16k
from utils import (
    documents_from_dir,
    estimate_cost_path,
)

"""
A smol design doc generator for any open source project
"""


def summarize_documents(documents):
    """Map-reduce summarize documents to a design doc"""
    llm = ChatOpenAI(temperature=1, model_name=GPT_3_5_TURBO_16k)
    reduce_llm = ChatOpenAI(temperature=1, model_name=GPT_4)

    map_prompt = """Give a one line summary of below with key functionality and components:
    {text}"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    collapse_prompt = """Give a concise summary of below with key functionality and components:
    {text}"""
    collapse_prompt_template = PromptTemplate(
        template=collapse_prompt, input_variables=["text"]
    )

    reduce_prompt = """
    Write a full technical design doc, in markdown format, for below codebase. 

    At a high level, discuss the purpose and functionalities of the codebase, major tech stack used, 
    and an overview of the architecture. Describe the framework and languages used for each tech 
    layer and corresponding communication protocols. If there's any design unique about this 
    codebase, make sure to discuss those aspect in closer detail. 

    Then in more details, describe the mission critical API endpoints. Describe the overall 
    user experience and product flow. Talk about the data storage and retrieval strategy, 
    including performance considerations and specific table schema. Touch on the deployment 
    flow and infrastructure set up. Include topics around scalability, fault tolerance and 
    monitoring.

    Lastly, briefly touch on the security and authentication aspect. Talk about potential 
    future improvements and enhancement to the feature set. 

    {text}
    """
    reduce_prompt_template = PromptTemplate(
        template=reduce_prompt, input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm,
        reduce_llm=reduce_llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        collapse_prompt=collapse_prompt_template,
        combine_prompt=reduce_prompt_template,
        return_intermediate_steps=True,
        verbose=True,
    )
    # track token and dollar usage
    with get_openai_callback() as usage_callback:
        res = chain({"input_documents": documents})
        print(usage_callback)

    design = res["output_text"]
    file_summaries = {
        doc.metadata["path"]: res["intermediate_steps"][i]
        for i, doc in enumerate(documents)
    }
    return design, file_summaries


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def save_txt(data, filepath):
    with open(filepath, "w") as file:
        file.write(data)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # TODO Rewrite this into CLI

    project = "media-server"
    path = "/Users/yuansongfeng/Desktop/dev/media-server"

    docs = documents_from_dir(path)
    estimated_cost = estimate_cost_path(path)

    confirmation = input(f"Estimated cost is ${estimated_cost}. Continue? [y/n] ")
    if confirmation == "y":
        design, summaries = summarize_documents(docs)
        os.makedirs(f"generated/{project}", exist_ok=True)
        save_txt(design, f"generated/{project}/design.txt")
        save_json(summaries, f"generated/{project}/summaries.json")
    else:
        print("Aborted.")
