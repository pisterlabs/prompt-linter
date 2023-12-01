import json
from typing import Tuple
import openai
from constants import *

def get_knowledge_file_options() -> Tuple[str]:
    return tuple([*KNOWLEDGE_FILEPATH.glob("*.txt")])

def convert_txt2jsonl(knowledge_jsonl:Path, knowledge_txt:Path) -> None:
    # Process generating knew knowldege file
    with open(knowledge_txt, encoding="utf8") as f:
        lines = [{"text": line} for line in f.read().splitlines() if line]

    # Convert to a list of JSON strings
    json_lines = [json.dumps(l) for l in lines]

    # Join lines and save to .jsonl file
    json_data = '\n'.join(json_lines)
    with open(knowledge_jsonl, 'w') as f:
        f.write(json_data)
        
def update_knowledge_base(knowledge_jsonl:Path, knowledge_available_online:Path):
    # Delete existing file from API
    if knowledge_available_online():
        openai.File.delete(openai.File.list()['data'][0]['id'])
        openai.File.list()

    # Open knowledge base
    openai.File.create(file=open(knowledge_jsonl, encoding="utf8"), purpose='answers')

def knowledge_available_online() -> bool:
    return len(openai.File.list()['data']) >= 1