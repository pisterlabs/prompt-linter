from openai import OpenAI

import gzip
import json
import os
import tqdm

SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EMBEDDING_MODEL = "text-embedding-ada-002"
MSG_FILE = 'msg.json'

def extract_messages() -> list[str]:
    with open(os.path.join(SERVER_DIR, "data/msg.json")) as file:
        data = json.load(file)
        messages = data["messages"]
        messages = [msg for msg in messages if msg['content']]
        cleaned_messages = [f"{msg['author']['name']} \"{msg['content']}\"" for msg in messages]
        return cleaned_messages
    
def get_embeddings(inp: list[str], batch: int=1000) -> list[list[float]]:
    client = OpenAI()
    i = 0
    outputs = []
    while i < len(inp):
        result = client.embeddings.create(input=inp[i:i+batch], model=EMBEDDING_MODEL)
        outputs += [x.embedding for x in result.data]
        i += batch
    assert len(outputs) == len(inp)
    return outputs

def write_jsonl(filename: str, data: list[dict]):
    assert filename.endswith(".jsonl.gz")
    with open(filename, "wb") as fp:
        with gzip.GzipFile(fileobj=fp, mode="wb") as gz:
            for x in tqdm.tqdm(data):
                gz.write((json.dumps(x) + "\n").encode("utf-8"))
    
def main():
    # Get embeddings
    messages = extract_messages()
    embeddings = get_embeddings(messages)

    # Save embeddings
    info = [
        {"message": msg, "embed": embed} 
        for msg, embed in zip(messages, embeddings)
    ]
    output_filename = os.path.join(SERVER_DIR, "msg-embeddings.jsonl.gz")
    write_jsonl(output_filename, info)

if __name__ == "__main__":
	main()
