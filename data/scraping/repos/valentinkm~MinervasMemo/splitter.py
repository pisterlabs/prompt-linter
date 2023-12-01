from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
import tiktoken

# Recursive splitting to consider different separators in generic text
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200, 
    separators=["\n\n", "\n", " ", ""],
    length_function = len
)

def load_transcript(raw_md):
    loader = UnstructuredMarkdownLoader(raw_md)
    data = loader.load()
    return data

def split_transcript(raw_md):
    data = load_transcript(raw_md)
    docs = r_splitter.split_documents(data)
    return docs