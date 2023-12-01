import pinecone
from dotenv import load_dotenv
import json
import logging
import math
import os
import hashlib
import openai
import sys
sys.path.append('../helpers')
import helpers.linguistics as linguistics
from InstructorEmbedding import INSTRUCTOR
load_dotenv("../.env")

model = INSTRUCTOR('hkunlp/instructor-base')

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

openai.api_key = OPENAI_API_KEY


index = pinecone.Index(PINECONE_INDEX)

"""
    createEmbedding(text)
    text: string
    
    returns: list
    
    returns the semantic embedding of the text
"""
def createEmbedding(text) -> list:
    logging.info(f"creating embedding")
    instruction = "make a semantic embedding of this for queries:"
    embeddings = model.encode([[instruction, text]]).tolist()[0]
    return embeddings


"""
    get_summary(text)
    text: string

    returns: {
        "title": string or None,
        "summary": string,
        "tags": list
    }

    uses chatgpt3.5 function calling api to summarize the text and extract tags
"""
def get_summary(text) -> dict:
    messages = [
        {"role": "user", "content": text},
    ]

    logging.info(f"getting summary")
    functions = [
        {
            "name": "extract_data",
            "description": "Extracts summary data from raw text, usually a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The substantial summary of the text. Usually around a paragraph but feel free to be more or less verbose. Keep the same tone, style, and voice as the original text.",
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the text. Usually the title of the web page, but feel free to change it to something more descriptive but very succinct."
                    },
                    "tags": {"type": "array",
                             "items": {"type": "string"},
                             "description": "A list of keywords that can be used to group key attributes of text. Usually ~5 keywords depending on length of raw text."},
                },
                "required": ["summary", "tags"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call={
            "name": "extract_data",
        },
    )
    response_message = response["choices"][0]["message"]

    function_args = json.loads(response_message["function_call"]["arguments"])

    logging.info(f"got summary: {function_args}")
    return {
        "title": function_args.get("title") or None,
        "summary": function_args.get("summary"),
        "tags": function_args.get("tags")
    }

"""
    putDoc(title, ptr, tags, content)
    title: string
    ptr: string
    tags: list
    content: string

    returns: {
        "id": string,
        "title": string,
        "ptr": string,
        "tags": list,
        "content": string,
        "error": string
    }

    puts the document into the brain
"""
def putDoc(title, ptr, tags, content):
    content = linguistics.recursive_summarization(content)

    # get metadata
    metadata = get_summary(f"{ptr}\n\n {title}\n {content}")
    summary = metadata["summary"]
    tags = metadata["tags"] + tags if metadata["tags"] is not None else tags
    title = metadata["title"] if metadata["title"] is not None and metadata["title"] != "" else title

    # create embedding
    embedding = createEmbedding(summary)

    # create id
    id = hashlib.sha256((ptr).encode()).hexdigest()
    
    # put into pinecone
    resp = index.upsert(
        vectors=[
            (
                id,
                embedding,
                {
                    "title": title,
                    "ptr": ptr,
                    "tags": tags,
                    "content": summary,
                    "raw_content": content[:1000]
                }
            )
        ],
        namespace="docs"
    )

    logging.info(f"put doc: {resp}")

    return {
        "id": id,
        "title": title,
        "ptr": ptr,
        "tags": tags,
        "content": summary,
        "error": None
    }
