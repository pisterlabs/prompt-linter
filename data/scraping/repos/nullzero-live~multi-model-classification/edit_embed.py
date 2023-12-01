"""Python file to serve as the frontend"""
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from sidebar import sidebar
import openai

def query_refiner(query):
    response = openai.Completion.create(
    
    api_key=os.environ['OPENAI_API_KEY'],
    model="text-davinci-003",
    prompt=f"You are a professional that specializes in scikit-learn. You will summarize the query with an insightful, deep analysis of the result. Maximum 2 paragraphs and highly technical with explanations of what each part means.\n\n Query is: {query}", #CONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    temperature=0.1,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']


def load_chain(query):
    openai.api_key=os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings()
    docsearch=Pinecone.from_existing_index(index_name, embeddings)
    
    docs = docsearch.similarity_search(query, k=1)
    output = query_refiner(docs[0].page_content.replace('\n',''))

    return output



