from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from uuid import uuid4
from tqdm.auto import tqdm
import openai
import purconfig



# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
     

def configureTextSplitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

def splitContentsIntoChunks():
    chunks = []
    for idx, record in enumerate(tqdm(data)):
        texts = text_splitter.split_text(record['text'])
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i,
            'url': record['url']
        } for i in range(len(texts))])  



    #In the response below res we will find a JSON-like object containing our new embeddings within the 'data' field.
    #res includes ['object', 'data', 'model', 'usage']
def purcreateembeddings(chunks):
    #limit input to 8k tokens
    res = openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], engine=embed_model
    )

    # no of data elements
    len(res[data])

    #length of structure of data
    len(res['data'][0]['embedding']), len(res['data'][1]['embedding'])
