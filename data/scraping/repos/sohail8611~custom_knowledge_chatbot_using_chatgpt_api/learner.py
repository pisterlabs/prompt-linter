
from openai.embeddings_utils import get_embedding


import json

def append_to_json_file(file_path, new_objects):
    # Read the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Append new objects to the existing data
    data.append(new_objects)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def split_text_into_chunks(file_path):
    chunks = []
    
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', ' ')  # Read the file and replace newlines with spaces
    
    current_chunk = ""
    words = text.split()  # Split the text into words
    
    for word in words:
        if len(current_chunk) + len(word) <= 800:
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + ' '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


chunks = split_text_into_chunks('ddotpy.txt')

for i in chunks:
    embd = get_embedding(i,engine='text-embedding-ada-002')
    obj = {
        'chunk_text': i,
        'embeddings':embd
    }

    append_to_json_file('knowledge.json',obj)
