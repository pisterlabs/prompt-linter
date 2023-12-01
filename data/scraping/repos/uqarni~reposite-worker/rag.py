import openai
import os
import re
import random
from datetime import datetime, timedelta
import random
import time

#similarity search
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

def find_examples(query, type, k=8):
    if type == 'taylor_RAG':
        full_file = 'RAG_examples/taylor.csv'
        col1 = 'RAG_examples/taylorcol1.csv'

    elif type == 'taylorNMQR_RAG':
        full_file = 'RAG_examples/taylorNMQR.csv'
        col1 = 'RAG_examples/taylorNMQRcol1.csv'
        
    loader = CSVLoader(file_path=col1)

    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)


    db = FAISS.from_documents(data, embeddings)
    examples = ''
    docs = db.similarity_search(query, k)
    df = pd.read_csv(full_file)
    i = 1
    for doc in docs:
        input_text = doc.page_content[14:]
        try:
            output = df.loc[df['User Message'] == input_text, 'Assistant Message'].iloc[0]
        except:
            print('found error for input')

        try:
            examples += f'Example {i}: \n\nLead Email: {input_text} \n\nTaylor Response: {output} \n\n'
        except:
            continue
        i += 1
    return examples
