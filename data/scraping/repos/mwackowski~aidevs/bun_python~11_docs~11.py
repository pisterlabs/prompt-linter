import json

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.text import TextLoader
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

### CD TO THE DIRECTORY OF THE FILE

loader = TextLoader("docs.md")
doc = loader.load()[0]
documents = [
    Document(page_content=content) for content in doc.page_content.split("\n\n")
]

print(documents)

chat = ChatOpenAI()
descriptions = []
for doc in documents:
    descriptions.append(
        chat.predict_messages(
            messages=[
                SystemMessage(
                    content="Describe the following document with one of the following keywords: Mateusz, Jakub, Adam. Return the keyword and nothing else."
                ),
                HumanMessage(content=f"Document: {doc.page_content}"),
            ]
        )
    )

for doc, desc in zip(documents, descriptions):
    doc.metadata["source"] = desc.content

documents_data = [
    {"page_content": doc.page_content, "metadata": {"source": doc.metadata["source"]}}
    for doc in documents
]

# Save the JSON data to a file locally
file_path = "documents.json"
with open(file_path, "w") as json_file:
    json.dump(documents_data, json_file, indent=2)
