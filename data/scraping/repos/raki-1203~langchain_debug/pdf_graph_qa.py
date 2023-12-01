import os
import re

import tiktoken
from tqdm import tqdm

from langchain import OpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.indexes import GraphIndexCreator
from langchain.text_splitter import CharacterTextSplitter

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID

if __name__ == '__main__':
    loader = PyMuPDFLoader('./TheLittlePrince.pdf')

    pages = loader.load_and_split()

    texts = []
    for page, doc in enumerate(pages):
        if page <= 1:
            continue
        text = doc.page_content
        text = re.sub(r'\d{1,}\n\d*|!{1}\n', '', text)
        text = text.replace('\n', '').strip()
        texts.append(text)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=0)

    splitted_texts = []
    for text in texts:
        splitted_texts.extend(text_splitter.split_text(text))

    for text in tqdm(splitted_texts):
        index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
        graph = index_creator.from_text(text)  # graph 가 각각 생겨서 이걸 합치지 않고서는 정보를 다 합칠 수 있을까?
        print(graph.get_triples())




