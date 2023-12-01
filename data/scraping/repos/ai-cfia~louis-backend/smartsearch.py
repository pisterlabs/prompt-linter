from langchain.tools import tool

from louis import actions

MAX_TOKENS = 3000

@tool
def SmartSearch(query: str) -> str:
    """
    Returns list of documents from inspection.canada.ca,
    the official website of the CFIA
    (Canadian Food Inspection Agency or Agence Canadienne d'Inspection des Aliments in french) based on
    semantic similarity to query"""
    documents = actions.smartsearch(query)
    paragraphs = []
    total_tokens = 0
    for doc in documents:
        total_tokens += doc['tokens_count']
        if total_tokens > MAX_TOKENS:
            break
        paragraph = f"{doc['title']} from {doc['url']} : {doc['content']}"
        paragraphs.append(paragraph)
    return "\n".join(paragraphs)