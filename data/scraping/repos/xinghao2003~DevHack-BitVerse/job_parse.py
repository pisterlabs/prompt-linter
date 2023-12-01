from config import client, temp_path
import json
from langchain.document_loaders import UnstructuredFileLoader

parsed_job_details = []


def extract_job_details(files, process):
    if (type(files) is not list):
        files = [files]

    global parsed_job_details

    for file in files:
        if (existance_check(file)):
            continue
        process.write(f"Reading data")

        loader = UnstructuredFileLoader(temp_path + file.name)
        docs = loader.load()
        content = docs[0].page_content

        process.write(f"Extracting data")

        response_json = parse_job(content)
        if (response_json == None):
            continue

        parsed_job_details.clear()

        response_json['file'] = file.name
        response_json['id'] = len(parsed_job_details) + 1

        parsed_job_details.append(response_json)

        process.write(f"Data extracted successfully")

    process.update(label="Completed", expanded=False, state="complete")

    return parsed_job_details


def existance_check(file):
    for job in parsed_job_details:
        if (job['file'] == file.name):
            return True


def load_system_prompt():
    loader = UnstructuredFileLoader("./prompts/job_parse_system.txt")
    docs = loader.load()
    content = docs[0].page_content
    return content


def parse_job(job):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": f"{load_system_prompt()}"},
                  {"role": "user", "content": f"{job}"}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    print(response)

    if (response.choices[0].message.content == None):
        return None

    response_json = json.loads(
        response.choices[0].message.content)

    return response_json
