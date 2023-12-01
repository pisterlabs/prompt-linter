import json
from  langchain.schema import Document
from typing import Iterable

# function to write a dict into json file
def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# function to read a json into a dict
def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

#from langchain github site: https://github.com/langchain-ai/langchain/issues/3016
def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
