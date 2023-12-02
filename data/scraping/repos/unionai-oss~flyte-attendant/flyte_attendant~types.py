"""Custom types."""

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from flytekit.types.file import FlyteFile
from langchain.docstore.document import Document


@dataclass_json
@dataclass
class FlyteDocument:
    page_filepath: FlyteFile
    metadata: dict

    def to_document(self) -> Document:
        with open(self.page_filepath) as f:
            page_content = f.read()
        return Document(page_content=page_content, metadata=self.metadata)
