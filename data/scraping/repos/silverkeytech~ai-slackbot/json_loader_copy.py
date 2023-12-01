# import json
# from pathlib import Path
# from typing import Callable, Dict, List, Optional, Union

# from langchain.docstore.document import Document
# from langchain.document_loaders.base import BaseLoader

# class JSONLoader(BaseLoader):
#     def __init__(
#         self,
#         file_path: Union[str, Path],
#         content_key: Optional[str] = None,
#         ):
#         self.file_path = Path(file_path).resolve()
#         self._content_key = content_key

#     # def create_documents(self, processed_json):
#     #     documents = []
#     #     for item in processed_json:
#     #         content = ''.join(item)
#     #         document = Document(page_content=content, metadata={})
#     #         documents.append(document)
#     #     return documents
    
#     # # def create_documents(processed_data):
#     # #     documents = []
#     # #     for item in processed_data:
#     # #         content = ''.join(item)
#     # #         document = Document(page_content=content, metadata={})
#     # #         documents.append(document)
#     # #     return documents
    
#     # def process_item(self, item, prefix=""):
#     #     if isinstance(item, dict):
#     #         result = []
#     #         for key, value in item.items():
#     #             new_prefix = f"{prefix}.{key}" if prefix else key
#     #             result.extend(self.process_item(value, new_prefix))
#     #         return result
#     #     elif isinstance(item, list):
#     #         result = []
#     #         for value in item:
#     #             result.extend(self.process_item(value, prefix))
#     #         return result
#     #     else:
#     #         return [f"{prefix}: {item}"]

#     # def process_json(self,data):
#     #     if isinstance(data, list):
#     #         processed_data = []
#     #         for item in data:
#     #             processed_data.extend(self.process_item(item))
#     #         return processed_data
#     #     elif isinstance(data, dict):
#     #         return self.process_item(data)
#     #     else:
#     #         return []
        
#     # def load(self) -> List[Document]:
#     #     """Load and return documents from the JSON file."""

#     #     docs = []
#     #     with open(self.file_path, 'r') as json_file:
#     #         try:
#     #             data = json.load(json_file)
#     #             processed_json = self.process_json(data)
#     #             docs = self.create_documents(processed_json)
#     #         except json.JSONDecodeError:
#     #             print("Error: Invalid JSON format in the file.")
#     #     return docs

#     # def load(self) -> List[Document]:
#     #     """Load and return documents from the JSON file."""

#     #     docs=[]
#     #     with open(self.file_path, 'r') as json_file:
#     #         try:
#     #             data = json.load(json_file)
#     #             processed_json = self.process_json(data)
#     #             docs = self.create_documents(processed_json)
#     #         except json.JSONDecodeError:
#     #             print("Error: Invalid JSON format in the file.")
#     #     return docs





import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class JSONLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: str,
        content_key: Optional[str] = None,
        metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
        text_content: bool = True,
    ):
        self.file_path = Path(file_path).resolve()
        self.jq_schema = json.loads(jq_schema)
        self.content_key = content_key
        self.metadata_func = metadata_func
        self.text_content = text_content

    def load(self) -> List[Document]:
        data = json.loads(self.file_path.read_text())

        # Perform some validation
        if self.content_key is not None:
            self._validate_content_key(data)

        docs = []
        for item in data:
            metadata = dict(
                source=str(self.file_path),
                seq_num=len(docs) + 1,
            )
            text = self._get_text(item, metadata)
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _get_text(self, item: Any, metadata: dict) -> str:
        """Convert item to string format"""
        if self.content_key is not None:
            content = item.get(self.content_key)
            if self.metadata_func is not None:
                # We pass in the metadata dict to the metadata_func
                # so that the user can customize the default metadata
                # based on the content of the JSON object.
                metadata = self.metadata_func(item, metadata)
        else:
            content = item

        if self.text_content and not isinstance(content, str):
            raise ValueError(
                f"Expected page_content is string, got {type(content)} instead. \
                    Set `text_content=False` if the desired input for \
                    `page_content` is not a string"
            )

        # In case the text is None, set it to an empty string
        elif isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content) if content else ""
        else:
            return str(content) if content is not None else ""

    def _validate_content_key(self, data: Any) -> None:
        """Check if content key is valid"""
        sample = data.first()
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict), \
                    so sample must be a dict but got `{type(sample)}`"
            )

        if sample.get(self.content_key) is None:
            raise ValueError(
                f"Expected the jq schema to result in a list of objects (dict) \
                    with the key `{self.content_key}`"
            )

        if self.metadata_func is not None:
            sample_metadata = self.metadata_func(sample, {})
            if not isinstance(sample_metadata, dict):
                raise ValueError(
                    f"Expected the metadata_func to return a dict but got \
                        `{type(sample_metadata)}`"
                )