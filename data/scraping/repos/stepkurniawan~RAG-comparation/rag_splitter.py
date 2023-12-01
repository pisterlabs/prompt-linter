"""
Ref: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/split_by_token
Splitting text has many forms. The simplest one is by characters. 
But sometimes you want to split by words, sentences, paragraphs, or other tokens.
Normally, language models are trained using token as a splitter. 
Example of token splitters:
- tiktoken
- spaCy
- SentenceTransformers
- NLTK
- HuggingFace tokenizer
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import tiktoken



# using tiktoken as tokenizer 

enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
def split_data_to_docs(data, chunk_size = 200):
    splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=chunk_size, 
                chunk_overlap=chunk_size*0.2,
                length_function=length_function,
                )

    all_splits = splitter.split_documents(data)

    return all_splits

