import json
from datetime import datetime

from langchain.tools import tool

from tools.summarizer import summarize_just_urls

# chromadb_client = chromadb.Client()
# collection = chromadb_client.create_collection("notes")

@tool
def add_notes(title: str, source: str, note: str) -> str:
    '''
    Adds a note to the notebook with a title, source and corresponding note text.
    A note is added only when the information is required to be stored, or some interesting information is shared.
    '''
    try:
        timestamp = datetime.now().strftime('%m/%d/%Y, %H:%M')
        header = f"{source} : {timestamp}"
        ## Add to todo list
        with open('store/notes.md', 'a') as f:
            f.write(f"\n\n# {title}\n")
            f.write(f"- {header}\n")
            f.write(summarize_just_urls(note))
        return "Note added."
    except Exception as e:
        return json.dumps({'error': str(e)})

# @tool
# def search_notes(query: str) -> str:
#     '''
#     Searches for a note in the notebook.
#     '''
#     try:
#         results = collection.query(
#             query_texts=[query],
#             n_results=2,
#             fields=['documents', 'metadatas', 'ids']
#         )
#         return json.dumps(results)
#     except Exception as e:
#         return json.dumps({'error': str(e)})
