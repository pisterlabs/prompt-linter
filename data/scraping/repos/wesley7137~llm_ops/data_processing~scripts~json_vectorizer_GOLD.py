import json
from pathlib import Path
from typing import List, Union
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class JSONLoader(BaseLoader):
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs = []
        with open(self.file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                for item in data:
                    page_content = item.get('page_content', '')
                    metadata = item.get('metadata', {})
                    doc = Document(page_content=page_content, metadata=metadata)
                    docs.append(doc)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in the file.")
        
        # Print each document
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"Page Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 50)  # prints a separator line

        return docs

# Usage:
file_path='D:\\PROJECTS\\AGENT_X\\OrchestrAI\\memory\\data\\conversations.json'
loader = JSONLoader(file_path=file_path)
loader.load()
