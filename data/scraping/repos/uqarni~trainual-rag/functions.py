import openai


#similarity search
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from supabase_client import get_prompt

articles = get_prompt('tracyRAG_articles')

def find_txt_examples(query, chunk_size, chunk_overlap, k):
    with open('kb.txt', 'w') as f:
        f.write(articles)

    loader = TextLoader('kb.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)
    docs = db.similarity_search(query, k=k)

    examples = ""

    for doc in docs:
       examples += '\n\n' + '---------------------' + '\n\n' + doc.page_content
    return examples


def responder(bot_prompt, examples, query):
    messages = []
    prompt = {
        'role': 'system',
        'content': bot_prompt + examples}
    user_input = {'role': 'user', 'content': query}

    messages.append(prompt)
    messages.append(user_input)

    response = openai.ChatCompletion.create(
        model = 'gpt-4',
        temperature = 0,
        messages = messages,
        max_tokens = 300
    )
    return response["choices"][0]["message"]["content"]
