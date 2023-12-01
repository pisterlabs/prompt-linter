import json
import re
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
 
def split_into_chunks(text):
    return re.split(r'==[^\n]*==', text)
  
def main():
               
    chunks_variable = []
 
    with open('Onlinehelp-en.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    result = [{"title": page["title"], "__text": page["revision"]["text"]["__text"]} for page in data["mediawiki"]["page"]]
    
    for page in result:
        text = page["title"] + page['__text']
        chunks = split_into_chunks(text)
        chunks_variable.extend(chunks)
    openai_key = os.environ.get('OPENAI_API_KEY', '')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    VectorStore = FAISS.from_texts(chunks_variable, embedding=embeddings)
    with open(f"online_help2.pkl", "wb") as f:
        pickle.dump(VectorStore, f)
 
 
if __name__ == '__main__':
    main()
 