from typing import List

from langchain.schema import Document


def execute(documents: List[Document]) -> str:
    formatted_documents: List[str] = []
    for d in documents:
        document_strings: List[str] = []
        if title := d.metadata.get("title"):
            document_strings.append(f"Title: {title}")
        if source := d.metadata.get("source"):
            document_strings.append(f"Source: {source}")
        document_strings.append("Content:\n" + d.page_content)
        formatted_documents.append("\n".join(document_strings))
    return "\n\n".join(formatted_documents)
