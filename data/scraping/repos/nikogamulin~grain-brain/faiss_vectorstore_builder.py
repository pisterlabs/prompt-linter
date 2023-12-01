import os
import glob
import pickle
from tqdm import tqdm
import utils

from langchain import PromptTemplate, LLMChain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import DirectoryLoader

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the parent folder
os.chdir(script_dir)

DOCUMENTS_FOLDER = 'data/raw/articles_txt'
FAISS_INDEX_FOLDER = 'data/articles_index_faiss'
FAISS_INDICES_FOLDER = 'data/articles_indices_faiss'
CHROMA_INDEX_FOLDER = 'data/articles_index_chroma'
FAILED_CHUNKS = 'data/failed_chunks/failed_chunks.pkl'
PROCESSED_ARTICLES = 'data/processed/articles.pkl'

    

# SCRIPT INFO:
# 
# This script allows you to create a vectorstore from a file and query it with a question (hard coded).
# 
# It shows how you could send questions to a GPT4All custom knowledge base and receive answers.
# 
# If you want a chat style interface using a similar custom knowledge base, you can use the custom_chatbot.py script provided.

# Setup 
gpt4all_path = './models/gpt4all-converted.bin' 
llama_path = './models/ggml-model-q4_0.bin' 

embeddings = LlamaCppEmbeddings(model_path=llama_path)
llm = GPT4All(model=gpt4all_path, verbose=True)

os.makedirs(FAISS_INDICES_FOLDER, exist_ok=True)
processed_indices = utils.list_subfolders(FAISS_INDICES_FOLDER)
loader = DirectoryLoader(DOCUMENTS_FOLDER, glob='**/*.txt', show_progress=True)
documents = loader.load()
unprocessed_documents = list(filter(lambda doc: os.path.splitext(str(os.path.basename(doc.metadata['source'])))[0] not in processed_indices, documents))

# article_titles = get_article_titles(DOCUMENTS_FOLDER)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(unprocessed_documents)
# filter out all the chunks that are shorter than 400 characters (most likely citations, titles, etc.)
chunked_documents = list(filter(lambda doc: len(doc.page_content) > 400, chunked_documents))
chunk_groups = list(set(map(lambda chunk: chunk.metadata['source'], chunked_documents)))
valid_document_chunks = []
processed_indices = []

if os.path.exists(FAILED_CHUNKS):
    # Load data from pickle file
    with open(FAILED_CHUNKS, 'rb') as f:
        failed_chunks = pickle.load(f)
else:
    # Create an empty list if pickle file does not exist
    os.makedirs(FAILED_CHUNKS, exist_ok=True)
    failed_chunks = []

failures_count = 0
with tqdm(total=len(chunk_groups)) as pbar:
    for chunk_group in chunk_groups:
        current_group = list(filter(lambda doc: doc.metadata['source'] == chunk_group, chunked_documents))
        indices_current = []
        current_group_count = len(current_group)
        for idx_chunk, document_chunk in enumerate(current_group):
            pbar.set_postfix_str(f'{idx_chunk+1}/{current_group_count}')
            try:
                db = FAISS.from_documents([document_chunk], embeddings)
                indices_current.append(db)
            except Exception as e:
                failed_chunks.append(document_chunk)
                failures_count += 1
                print(e)
        db_merged = indices_current[0]
        for i in range(1, len(indices_current)):
            db_merged.merge_from(indices_current[i])
        filename_with_extension = os.path.basename(chunk_group)
        db_merged.save_local(os.path.join(FAISS_INDICES_FOLDER, os.path.splitext(filename_with_extension)[0]))
        pbar.update(1)
    
# merge all indices
processed_indices = utils.list_subfolders(FAISS_INDICES_FOLDER)
index_merged = FAISS.load_local(os.path.join(FAISS_INDICES_FOLDER, processed_indices[0]), embeddings)
for i in range(1, len(processed_indices)):
    index_merged.merge_from(FAISS.load_local(os.path.join(FAISS_INDICES_FOLDER, processed_indices[i]), embeddings))
index_merged.save_local(FAISS_INDEX_FOLDER)

if failures_count > 0:
    with open(FAILED_CHUNKS, 'wb') as f:
        pickle.dump(failed_chunks, f)
    

# test the index
index = FAISS.load_local(FAISS_INDEX_FOLDER, embeddings)

# Set your query here manually
question = "how does carbon affect nitrogen mineralization?"
matched_docs, sources = utils.similarity_search(question, index)

template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

context = "\n".join([doc.page_content for doc in matched_docs])
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))