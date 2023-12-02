import openai
import pinecone
import numpy as np
import pandas as pd
import pickle
import tiktoken
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pinecone


#initialize openai
openai.organization = 'org-kbmIwXuUVgonTktQMnQiwfrt'
openai.api_key = "''"

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_text(pdf, starting_page, ending_page):
    reader = PdfReader(pdf)
    number_of_pages = len(reader.pages)
    extractedText = ""
    for i in range(starting_page-1, ending_page):
        page = reader.pages[i]
        extractedText += page.extract_text() 
    return extractedText

def get_documents(pdf, starting_page, ending_page):
    text = get_text(pdf, starting_page, ending_page)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0, separator='.')
    docs = text_splitter.split_text(text)
    return docs

def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]['embedding']


# Get the Documents
docs = get_documents('ex2.pdf', 146, 180)
print('Parsed Documents')
# Create A PineCone Index
index_name = 'openai-ex212'
pinecone.init(
    api_key="''",
    environment="us-east1-gcp"
)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(get_embedding(docs[0])),
        metric='cosine',
        # metadata_config={'indexed': ['channel_id', 'published']}
    )
    print('Created Index')

    index = pinecone.Index(index_name)
    # Create Embeddings for each
    to_upsert = []
    i = 0
    for doc in docs:
        i+=1
        data  = get_embedding(doc)
        to_upsert.append((str(i), data, {'text': doc}))

        print(i)
    index.upsert(vectors=to_upsert)
    print('Populated Index')





def get_answer(prompt):
    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    return response["choices"][0]["text"].strip(" \n")


# check if index already exists (it shouldn't if this is first time)

# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())

while True:
    user_input = input("Enter your input: ")
    input_embedded = get_embedding(user_input)
    res = index.query(input_embedded, top_k=2, include_metadata=True)
    context = ""
    for match in res['matches']:
        context += match['metadata']['text'] + "\n";
    
    prompt = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I dont know" and then continue the answer.\n\nContext:\n"""
    print(prompt)
    prompt += context
    prompt += f'\n\n{user_input}' + "\nA:"
    # print(prompt)
    print(get_answer(prompt))



# res = openai.Embedding.create(
#     input=[
#         "Sample document text goes here",
#         "there will be several phrases in each batch"
#     ], engine=MODEL
# )
# embeds = [record['embedding'] for record in res['data']]

# #initialize pinecone
# pinecone.init(
#     api_key="''",
#     environment="us-east1-gcp"
# )

# # check if 'openai' index already exists (only create index if not)
# if 'openai' not in pinecone.list_indexes():
#     pinecone.create_index('openai', dimension=len(embeds[0]))
# # connect to index
# index = pinecone.Index('openai')

# print(res)
# import pdb
# pdb.set_trace()