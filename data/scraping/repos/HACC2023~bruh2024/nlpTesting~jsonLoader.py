import json
from langchain.docstore.document import Document


def load_json_file(file_path):
    docs = []
    # Load JSON file
    with open(file_path, encoding="iso-8859-1") as file:
        data = json.load(file)

    # Iterate through 'pages'
    for index, question in data.items():
        q = question['question']
        a = question['answer']
        docs.append(Document(page_content=a, metadata={index: q}))
    return docs
