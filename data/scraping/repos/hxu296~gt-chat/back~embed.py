import os
import os.path as osp
from typing import List
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores.faiss import FAISS
import pandas as pd
import nltk
nltk.download('punkt')

PROCESSED_CSV_DIRECTORY = "processed" # Directory to save processed CSV file

def create_docs() -> List[Document]:
    docs = []
    df = pd.read_csv(osp.join(PROCESSED_CSV_DIRECTORY, 'scraped.csv'))
    for index, row in df.iterrows():
        doc = Document(page_content=row['text'], metadata={"source": row['url']})
        docs.append(doc)
    return docs


docs = create_docs()

doc_chunks = []
seen_chunks = set()
total_websites = set()
total_words = 0
splitter = NLTKTextSplitter(chunk_size=1024)
for source in tqdm(docs):
    for chunk in splitter.split_text(source.page_content):
        if chunk not in seen_chunks:
            doc_chunks.append(
                Document(page_content=chunk, metadata=source.metadata))
            total_words += len(chunk.split())
            total_websites.add(source.metadata['source'])
        seen_chunks.add(chunk)

print(f'Total websites: {len(total_websites)}')
print(f'Total chunks: {len(doc_chunks)}')
print(f'Total words: {total_words}')
print(f'Avg words per chunk: {int(total_words / len(doc_chunks))}')
print(f'Estimated embedding cost: ${total_words / 0.75 / 1000 * 0.0004:.2f}')

search_index = FAISS.from_documents(doc_chunks, OpenAIEmbeddings(model='text-embedding-ada-002'))

# persistent search index
search_index.save_local("search_index")
