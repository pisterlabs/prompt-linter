from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')
PINECONE_API_KEY = config('PINECONE_API_KEY')
PINECONE_API_ENV = config('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

def _similarity_search(input_query):
# initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "missioni"
    index = pinecone.Index(index_name)

    emb_query = embeddings.embed_query(input_query)

    #Search for similarity between query emb. and docs emb.
    res_vec = index.query(
    vector = emb_query,
    top_k = 3,
    include_metadata = True
    )
    docs = []
    for res in res_vec["matches"]:
        metadata = res["metadata"]
        text = metadata["text"]
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def _run(input_query):
    docs = _similarity_search(input_query)
    res = chain.run(input_documents=docs, question=input_query)
    return res

# def main():
#     input_q = ''
#     while input_q.lower() != 'exit':
#         print("Digita la query o 'exit' per uscire")
#         input_q = input("> ")
#         query = input_q
#         docs = _similarity_search(query)
#         res = _run(query, docs)
#         print(f">> {res} \n")

# main()