# First openai-api client
# using openai api with redis as a memory store
# memory store not used in main. 

from typing import Dict, List, Tuple, Any
from pprint import pprint
import os
import json
import openai
import tiktoken

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

def memory_init(vector_dimensions: int):
    DOC_PREFIX = "doc:"
    INDEX_NAME = "openai"
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except:
        # schema
        schema = (
            TagField("tag"),                       # Tag Field Name
            VectorField("vector",                  # Vector Field Name
                "FLAT", {                          # Vector Index Type: FLAT or HNSW
                    "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                    "DIM": vector_dimensions,      # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                }
            ),
        )

        # index Definition
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

        # create Index
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)

# query redis NHSW index for similar questions
# return top N results
def memory_query(
):
    query = (
        Query("(@tag:{ openai })=>[KNN 4 @vector $vec as score]")
         .sort_by("score")
         .return_fields("content", "tag", "score")
         .paging(0, 4)
         .dialect(2)
    )

    query_params = {"vec": query_embedding.tobytes()}
    r.ft(INDEX_NAME).search(query, query_params).docs

def memory_add():
    response = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")
    embeddings = np.array([r["embedding"] for r in response["data"]], dtype=np.float32)

    # Write to Redis
    pipe = r.pipeline()
    for i, embedding in enumerate(embeddings):
        pipe.hset(f"doc:{i}", mapping = {
            "vector": embedding.tobytes(),
            "content": texts[i],
            "tag": "openai"
        })
    res = pipe.execute()

    pass

def configure_chat(
) -> Tuple[
    Dict[str, Any], List[Dict[str, str]]
]:
    with open("config.json") as f:
        config = json.load(f)
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    chat = [{"role": "system", "content": config["chat"]["system"]}]
    return config, chat

def count_tokens(
    chat: List[Dict[str, str]],
    config: Dict[str, Any]
) -> int:
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    model = config["gpt"]["model"]
    encoding = tiktoken.encoding_for_model(model)
    for message in chat:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def ask_question(
        config, 
        chat: List[Dict[str, str]], 
        bias: Dict[str, float] = {}
) -> List[
    Dict[str, str]
]:
    config_gpt = config["gpt"]
    pprint(chat)
    response = openai.ChatCompletion.create(
        messages = chat,
        logit_bias = bias,
        stream = True,
        **config_gpt
    )

    full_response = ""
    for chunk in response:
        try:
            chunk_content = chunk["choices"][0]["delta"]["content"]
            full_response += chunk_content
            print(chunk_content, end='')
        except Exception:
            pass
    print()
    chat.append({'role': 'assistant', 'content': full_response})

    return chat

def compress_chat(
    config: Dict[str, Any],
    chat: List[Dict[str, str]]
) -> List[
    Dict[str, str]
]:
    config_chat = config["chat"]
    # Simple compression: just remove the beginning of the chat
    if len(chat) > config_chat["keep_messages"]:
        chat = (
            [chat[0]] +
            chat[1 - config_chat["keep_messages"]:]
        )
    return chat

def get_user_input(
    config: Dict[str, Any],
    chat: List[Dict[str, str]]
) -> Tuple[
    List[Dict[str, str]], 
    bool
]:
    config_chat = config["chat"]
    name = config_chat["name"]
    num_tokens = count_tokens(chat, config)
    print(f"num_tokens: {num_tokens}")
    new_question = input("question >>> ")
    chat.append({
        'role': 'user',
        'name': name,
        'content': new_question
    })
    if new_question == config_chat["end_message"]:
        return chat, True 

    return chat, False


def main():
    config, chat = configure_chat()
    chat, finish = get_user_input(config, chat)
    while not finish:
        chat = compress_chat(config, chat)
        chat = ask_question(config, chat)
        chat, finish = get_user_input(config, chat)


if __name__ == '__main__':
    main()
