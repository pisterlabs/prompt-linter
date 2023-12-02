from tqdm.auto import tqdm

import os
import pinecone
import yaml

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

f = open('params/credentials.yml', 'r')
credential = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', credential['OPENAI_API_KEY'])
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', credential['PINECONE_API_KEY'])
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', credential['PINECONE_API_ENV'])


def user_manual_search(question):
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002',openai_api_key=OPENAI_API_KEY)

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    # switch back to normal index for langchain
    index = pinecone.Index(index_name = credential['PINECONE_INDEX'])

    text_field = "text"
    vectorstore = Pinecone(index, embeddings.embed_query, text_field)
    vectors = vectorstore.similarity_search(question)

    llm = OpenAI(temperature=0, model_name='text-davinci-003', openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=vectors, question=question)
    
    return result.strip()