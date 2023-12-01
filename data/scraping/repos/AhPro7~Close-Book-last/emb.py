import os
import google.generativeai as palm
from tqdm import tqdm
palm.configure(api_key=os.environ["GOOGLE_API_KEY"])
import pandas as pd

import google.generativeai as palm

import chromadb
from chromadb.api.types import Documents, Embeddings

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
from langchain.text_splitter import CharacterTextSplitter #text splitter

import textwrap


#models

# embedText
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]



# some functions
def make_embeddings(text):
    return palm.generate_embeddings(model=model, text=text)['embedding']



# we will embed the whole documents in directory




books_pth = 'Books'
books = os.listdir(books_pth)
# print(books) to select the book by index
for i in range(len(books)):
    print(i, books[i])

# select the book
book_index = int(input('Select the book index: '))
book = books[book_index]
book_name = book
book_pth = os.path.join(books_pth, book)
print('Selected book:', book)
print()

# check if the book is already in the database(Embeddings folder)

if (book+'.csv') in os.listdir('Embeddings'):
    print('The book is already in the database')
    exit(0)

# get the pdf files in the book
pdfs = os.listdir(book_pth)
pdf_list = []

for pdf in pdfs:
    book = PyPDFLoader(os.path.join(book_pth, pdf))
    text_in_book = book.load()

    for page in tqdm(text_in_book):
        pdf_list.append(page.page_content)

    print('Successfully loaded', pdf)
    print()


df = pd.DataFrame(pdf_list)
df.columns = ['Text']
df['chars_len'] = df['Text'].apply(lambda x: len(x))
df = df[df['chars_len'] > 10] # change the number to filter the text to remove the blank pages

df['Embeddings'] = df['Text'].apply(make_embeddings)

df.to_csv(os.path.join('Embeddings', book_name+'.csv'), index=False)

print('Successfully embedded', book_name)