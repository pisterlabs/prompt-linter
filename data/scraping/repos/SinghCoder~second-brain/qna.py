from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import VectorstoreIndexCreator

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def qna(q):
    loader = UnstructuredMarkdownLoader("store/notes.md")
    index = VectorstoreIndexCreator().from_loaders([loader])

    output = index.query(q)
    return output
