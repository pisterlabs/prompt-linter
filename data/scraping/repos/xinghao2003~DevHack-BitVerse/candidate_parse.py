from config import client, temp_path
import json
from langchain.document_loaders import UnstructuredFileLoader

parsed_candidates = []


def add_candidates_to_list(files, process):
    if (type(files) is not list):
        files = [files]

    global parsed_candidates

    for file in files:
        if (existance_check(file)):
            continue

        process.write(f"Reading data")

        loader = UnstructuredFileLoader(temp_path + file.name)
        docs = loader.load()
        content = docs[0].page_content

        process.write(f"Extracting data")

        response_json = parse_candidate(content)

        if (response_json == None):
            continue

        response_json['file'] = file.name
        response_json['id'] = len(parsed_candidates) + 1
        parsed_candidates.append(response_json)

        process.write(f"Data extracted successfully")

    process.update(label="Completed", expanded=False, state="complete")

    return parsed_candidates


def remove_from_list(file):
    global parsed_candidates
    for candidate in parsed_candidates:
        if (candidate['file'] == file.name):
            continue
        parsed_candidates.remove(candidate)


def existance_check(file):
    for candidate in parsed_candidates:
        if (candidate['file'] == file.name):
            return True


def load_system_prompt():
    loader = UnstructuredFileLoader("./prompts/candidate_parse_system.txt")
    docs = loader.load()
    content = docs[0].page_content
    return content


def parse_candidate(candidate):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": f"{load_system_prompt()}"},
                  {"role": "user", "content": f"{candidate}"}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    print(response)

    if (response.choices[0].message.content == None):
        return None

    response_json = json.loads(
        response.choices[0].message.content)

    return response_json
