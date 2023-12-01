import os
from typing import Optional
import json
import uuid
from pathlib import Path

from pydantic import BaseModel
import openai
import torch
from transformers import BertTokenizerFast, BertModel
import chromadb
from chromadb.config import Settings
import cohere
from fastapi import FastAPI
# import modal

from translate import translate_text


# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "chromadb",
#         "fastapi",
#         "pydantic",
#         "openai==0.27.2",
#         "torch",
#         "transformers",
#         "google-cloud-translate",
#         "cohere",
#     ).copy(
#         mount=modal.Mount.from_local_file(
#             local_path=Path("iso639-1.json"), remote_path=Path('iso639-1.json')
#         ),
#     ).copy(
#         mount=modal.Mount.from_local_dir(
#             local_path=Path(".chromadb/"), remote_path=Path('.chromadb/')
#         ),
#     )
# )
# stub = modal.Stub("chatTRE-api-server", image=image)


app = FastAPI()


# @stub.function()
# @modal.asgi_app()
# def fastapi_app():


llm = 'chatgpt'
# llm = 'cohere'
embeddings = None  # Use default chromadb embeddings
# embeddings = 'labse'  # Use labse embeddings

# llm API key setup
if llm == 'cohere':
    co = cohere.Client(os.environ["COHERE_KEY"])
elif llm == 'chatgpt':
    openai.api_key = os.environ.get("OPENAI_KEY")

if embeddings and embeddings.lower() == 'labse':
    cache_path = 'bert_cache/'
    tokenizer = BertTokenizerFast.from_pretrained('setu4993/LaBSE', cache_dir=cache_path)
    model = BertModel.from_pretrained('setu4993/LaBSE', cache_dir=cache_path).eval()

with open('iso639-1.json') as f:
    iso_639_1 = json.load(f)

# Vector store (assuming the .chromadb directory already exists. If not, run db.py first)
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chromadb" 
))

if embeddings and embeddings.lower() == 'labse':
    collection = client.get_collection("tyndale-labse")
else:
    collection = client.get_collection("tyndale")


state_dict = {}

# @stub.function()
def get_embeddings(query, tokenizer, model):  # Only needed if using labse embeddings
    query_input = tokenizer(query, return_tensors="pt", padding=False, truncation=True)
    with torch.no_grad():
        query_output = model(**query_input)
    embedding = query_output.pooler_output.tolist()[0]
    return embedding


# @stub.function()
def add_text(text, state):
    query_text = '\n'.join([x[0] + '/n' + x[1][:50] + '\n' for x in state]) + text  # Add the previous queries and answers to the search query
    print(f'{query_text=}')

    translation_response = translate_text(query_text)
    english_query_text = translation_response.translations[0].translated_text
    query_language_code = translation_response.translations[0].detected_language_code
    query_language = iso_639_1[query_language_code]
    print(f'{query_language=}')
    print(f'{english_query_text=}')
    # Get the context from chroma
    if embeddings:
        query_embeddings = get_embeddings(query_text, tokenizer, model)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=10
        )
    else:  # Use default chromadb embeddings
        results = collection.query(
            query_texts=[english_query_text],
            n_results=10
        )

    # Prompt.
    context = '['
    for i in range(len(results['documents'][0])):
        print(results['metadatas'][0][i])
        context += "{source:" + results['metadatas'][0][i]['citation'] + ', text: ' + results['documents'][0][i] + '}' + ','
    context += ']' + '\n'
    print(f'{context=}')

    # Construct prompt.
    chat_prefix = "The following is a conversation with an AI assistant for Bible translators. The assistant is"
    chat_prefix += f" helpful, creative, clever, and very friendly. The assistant only responds in the {query_language} language.\n"
    prompt = (
        chat_prefix +
        f'Read the paragraph below and answer the question, using only the information in the context delimited by triple backticks. Answer only in the {query_language} language. '
        f'At the end of your answer, include the source of each context text that you used. You may use more than one, and include the sources of all those you used. '
        # f' Respond in the following format:' + '{' +
        # '"answer":<answer>, "sources": [<keys>]' + '}' + 
        
        f'If the question cannot be answered based on the context alone, write "Sorry i had trouble answering this question, based on the information i found\n'
        f"\n"
        f"Context:\n"
        f"```{ context }```\n"
        f"\n"
    )

    if len(state) > 0:
        if len(state) > 3:
            trim_state = state[-3:]
        else:
            trim_state = state
        for exchange in trim_state:
            prompt += "\nHuman: " + exchange[0] + "\nAI: " + exchange[1]
        prompt += "\nHuman: " + text + "\nAI: "
    else:
        prompt += "\nHuman: " + text + "\nAI: "
    print(f'{prompt=}')
    
    if llm == 'cohere':
        # Get the completion from co:here.
        response = co.generate(model='xlarge',
                            prompt=prompt,
                            max_tokens=200,
                            temperature=0)
        answer = response.generations[0].text

    elif llm == 'chatgpt':
        #ChatGPT reponse
        response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
        answer = response['choices'][0]["message"]["content"]
    else:
        print("No LLM specified")
        return '', state
    
    print(f'{answer=}')

    state.append((text, answer))
    return answer, state

class TextIn(BaseModel):
    text: str
    chat_id: Optional[str] = None


class TextOut(BaseModel):
    text: str
    chat_id: str


# @stub.function()
@app.post("/ask", response_model=TextOut)
def ask(input: TextIn):
    print(f'{input=}')
    if input.chat_id is None or input.chat_id == '':
        input.chat_id = str(uuid.uuid4())
        state_dict[input.chat_id] = []

    text, state_dict[input.chat_id] = add_text(input.text, state_dict.get(input.chat_id, []))
    print(f'{text=}')
    print(f'{state_dict[input.chat_id]=}')
    return {'text': text, 'chat_id': input.chat_id}

