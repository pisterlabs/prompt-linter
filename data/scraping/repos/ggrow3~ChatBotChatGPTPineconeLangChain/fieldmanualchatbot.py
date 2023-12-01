import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import os
import API_KEYS


pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  
    environment=os.environ['PINECONE_API_ENV']  
)
index_name = "fieldmanual"

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
pinecone = Pinecone.from_existing_index(index_name,embeddings)

openAI = OpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])

chain = load_qa_chain(openAI, chain_type="stuff")


def askGPT(prompt):
    docs = pinecone.similarity_search(prompt, include_metadata=True)
    ch = chain.run(input_documents=docs, question=prompt)
    print(ch)

def main():
    while True:
        print('Open AI + Pinecone: Field Manual Querying\n')
        prompt = "prompt:" + input()
        
        askGPT(prompt)
        print('\n')
main()

