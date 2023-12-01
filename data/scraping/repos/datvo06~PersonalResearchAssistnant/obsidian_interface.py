from langchain.document_loaders import ObsidianLoader
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from vectorstore import VectorstoreIndexCreator
from langchain.callbacks import get_openai_callback
import os
import pickle as pkl
import time
from langchain import OpenAI
from llm_utils import get_gpt4_llm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from settings import OBSIDIAN_PATH, OPENAI_API_KEY


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

loader = ObsidianLoader(OBSIDIAN_PATH)
embeddings = OpenAIEmbeddings()

documents = loader.load()
obsidian_db_path = 'obsidian_db'

if os.path.exists(obsidian_db_path):
    last_timestamp = os.path.getmtime('last_timestamp.pkl')
    index = VectorstoreIndexCreator().from_persistent_index(obsidian_db_path)
    db = index.vectorstore
else:
    index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embeddings, vectorstore_kwargs={
                                    "persist_directory": obsidian_db_path}).from_loaders([loader])
    index.vectorstore.persist()
    path2id = {doc.metadata['path'][len(OBSIDIAN_PATH):]: i for (
        i, doc) in enumerate(documents)}
    last_timestamp = time.time()
    pkl.dump(last_timestamp, open('last_timestamp.pkl', 'wb'))
    db = index.vectorstore


def update():
    global db
    global path2id
    global last_timestamp
    documents = loader.load()
    # gather all new doc that is created or mofiied after last_timestamp
    new_docs = [doc for doc in documents if doc.metadata['path']
        [len(OBSIDIAN_PATH):] not in path2id]
    modified_docs = [doc.metadata['last_modified'] > last_timestamp and doc.metadata['path'][len(
        OBSIDIAN_PATH):] in path2id for doc in documents]
    removed_docs = [doc.metadata['path'][len(
        OBSIDIAN_PATH):] in path2id and doc.metadata['last_modified'] > last_timestamp for doc in documents]

    # first, update the modified docs
    for doc in modified_docs:
        doc_id = path2id[doc.metadata['path'][len(OBSIDIAN_PATH):]]
        # Update doc: todo


def retrieve_docs(query, db, top_k=10):
    results = db.similarity_search(query, top_k=top_k)
    return results


def get_generate_prompt_template():
    prompt_template = """Use the context below to write a 400 word blog post about the topic below:
        Context: {context}
        Topic: {topic}
        Blog post:"""
    return PromptTemplate(prompt_template, input_variables=['context', 'topic'])


def summarize_arxiv(link: str, output_path=None):
    '''Summarize an arxiv paper and output to a file'''


def handle_critical(qa_critical, query_critical):
    query=query_critical[len("[CRITICAL]"):].strip()
    results=qa_critical(query)
    return results




if __name__ == '__main__':
    llm_normal=OpenAI()
    llm_critical=get_gpt4_llm()
    retriever = db.as_retriever()
    retriever_critical = db.as_retriever()
    retriever_critical.search_kwargs = {"k": 20}
    qa_critical=RetrievalQAWithSourcesChain.from_chain_type(
        llm_critical, chain_type = "stuff", retriever=retriever_critical)
    while(True):
        query=input("Enter query: ")
        if query.startswith("[CRITICAL]"):
            '''
            # First, retrieve the doc
            doc_results = retrieve_docs(query[len("[CRITICAL]"):].strip(), db)
            # print the result and exit first
            for doc in doc_results:
                print(doc)
            exit()
            '''
            with get_openai_callback() as cb:
                results=handle_critical(qa_critical, query)
                print("\n Answer:", results['answer'])
                print("The sources are from the following files: ",
                      results['sources'])
                print("tokens used: ", cb.total_tokens)
        else:
            llm=llm_normal
            with get_openai_callback() as cb:
                result=index.query_with_sources(query, llm = llm_normal)
                print("\n Answer:", result['answer'])
                print("The sources are from the following files: ",
                      result['sources'])
                print("tokens used: ", cb.total_tokens)
        print("===============================\n")
