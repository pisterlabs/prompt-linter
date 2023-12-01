import pickle as pkl
import json
from vectorstore import VectorstoreIndexCreator
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from llm_utils import get_gpt4_llm, get_gpt35_turbo_llm
from settings import PDF_DICT_PATH, PDF_DB_DIR, PDF_RESULT_PATH, PDF_RESULT_DIR, OBSIDIAN_PATH, PDF_RESULT_DIR_LIGHT, PDF_RESULT_PATH_LIGHT
from langchain.prompts import PromptTemplate
import uuid
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import LLMChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import sys
import argparse

PDF_DICT = None
PDF_RESULT_DICT = None
PDF_RESULT_DICT_LIGHT = None


def load_dict():
    global PDF_DICT
    if PDF_DICT is None:
        try:
            pdf_dict = json.load(open(PDF_DICT_PATH, 'r'))
        except:
            pdf_dict = {}
        PDF_DICT = pdf_dict
    return PDF_DICT


def save_dict(pdf_dict):
    json.dump(pdf_dict, open(PDF_DICT_PATH, 'w'))


def load_result_dict(light=False):
    global PDF_RESULT_DICT
    global PDF_RESULT_DICT_LIGHT
    if not light:
        if PDF_RESULT_DICT is None:
            try:
                pdf_result_dict = json.load(open(PDF_RESULT_PATH, 'r'))
            except:
                pdf_result_dict = {}
            PDF_RESULT_DICT = pdf_result_dict
        return PDF_RESULT_DICT
    else:
        if PDF_RESULT_DICT_LIGHT is None:
            try:
                pdf_result_dict = json.load(open(PDF_RESULT_PATH_LIGHT, 'r'))
            except:
                pdf_result_dict = {}
            PDF_RESULT_DICT_LIGHT = pdf_result_dict
        return PDF_RESULT_DICT_LIGHT


def save_result_dict(pdf_result_dict, light=False):
    if not light:
        json.dump(pdf_result_dict, open(PDF_RESULT_PATH, 'w'))
    else:
        json.dump(pdf_result_dict, open(PDF_RESULT_PATH_LIGHT, 'w'))


def create_or_get_pdf_db(pdf_file: str, pdf_dict: dict = None):
    if pdf_dict is None:
        pdf_dict = load_dict()
    os.makedirs(PDF_DB_DIR, exist_ok=True)
    # if this pdf file is already in the db, return the persistence
    if pdf_file in pdf_dict:
        db_path = pdf_dict[pdf_file]
        index = VectorstoreIndexCreator().from_persistent_index(
            pdf_dict[pdf_file])
    else:
        # create a new db with random unique name
        db_path = f"{PDF_DB_DIR}/" + str(uuid.uuid4())
        pdf_dict[pdf_file] = db_path
        while db_path in pdf_dict.values():
            db_path = f'{PDF_DB_DIR}/' + str(uuid.uuid4())
        # create the db
        loader = PyMuPDFLoader(pdf_file)
        index = VectorstoreIndexCreator(vectorstore_cls=Chroma,
                                        embedding=OpenAIEmbeddings(),
                                        vectorstore_kwargs={
                                            "persist_directory": db_path
                                        }).from_loaders([loader])
        index.vectorstore.persist()
        save_dict(pdf_dict)
    return index


def get_default_paper_query() -> List[str]:
    return [
        'What is the main contribution of this paper?',
        'How does this paper compare to previous work?',
        'What is the main methodology of the paper, formally?',
        'What is the main dataset used in this paper?',
        'What is the experiment settings of this paper?',
        'What is the main results of this paper?',
        'What is the main limitation of this paper?',
        'What is the main future work of this paper?',
        'Pose 5 questions that you would ask the authors of this paper that is not mentioned in this paper.',
        'Critique this paper.'
    ]


def get_default_paper_prompt() -> PromptTemplate:
    questions = get_default_paper_query()
    joined_question = "\n".join(
        [f"{i}. {q}" for i, q in zip(range(1,
                                           len(questions) + 1), questions)])
    refine_template = """
You job is to produce a final answer
We have provided an existing answer up to a certain point: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below
--------------
{text}
--------------
Given the new context, refine the original answers to the following questions:

""" + joined_question + """
If the context isn't useful, return the original answers."""
    refine_template = PromptTemplate(
        input_variables=["existing_answer", "text"], template=refine_template)
    return refine_template


def query_pdf_summarize_default(pdf_file: str):
    os.makedirs(PDF_RESULT_DIR, exist_ok=True)
    pdf_result_dict = load_result_dict()
    if pdf_file in pdf_result_dict:
        try:
            return json.load(
                open(pdf_result_dict[pdf_file], 'r'))
        except:
            pdf_result_dict.pop(pdf_file)
    refine_template = get_default_paper_prompt()
    chain = load_summarize_chain(get_gpt4_llm(),
                                 chain_type="refine",
                                 verbose=False,
                                 refine_prompt=refine_template)
    docs = PyMuPDFLoader(pdf_file).load()
    result_path = f"{PDF_RESULT_DIR}/" + str(uuid.uuid4())
    while result_path in pdf_result_dict.values():
        result_path = f'{PDF_RESULT_DIR}/' + str(uuid.uuid4())
    pdf_result_dict[pdf_file] = result_path
    result = chain({"input_documents": docs}, return_only_outputs=True)
    json.dump(result, open(result_path, 'w'))
    save_result_dict(pdf_result_dict)
    return result


def query_pdf_summarize(pdf_file: str):
    os.makedirs(PDF_RESULT_DIR_LIGHT, exist_ok=True)
    pdf_result_dict = load_result_dict(light=True)
    if pdf_file in pdf_result_dict:
        try:
            return json.load(
                open(pdf_result_dict[pdf_file],
                     'r'))
        except:
            pdf_result_dict.pop(pdf_file)
    refine_template = get_default_paper_prompt()

    chain = load_summarize_chain(get_gpt35_turbo_llm(),
                                 chain_type="refine",
                                 verbose=False,
                                 refine_prompt=refine_template)
    docs = PyMuPDFLoader(pdf_file).load()
    recursive_character_text_splitter = (
        RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=3000,
            chunk_overlap=0,
        ))
    docs = recursive_character_text_splitter.split_documents(docs)

    result_path = f"{PDF_RESULT_DIR_LIGHT}/" + str(uuid.uuid4())
    while result_path in pdf_result_dict.values():
        result_path = f'{PDF_RESULT_DIR_LIGHT}/' + str(uuid.uuid4())
    pdf_result_dict[pdf_file] = result_path
    result = chain({"input_documents": docs}, return_only_outputs=True, )
    json.dump(result, open(result_path, 'w'))
    save_result_dict(pdf_result_dict, light=True)
    return result


def query_pdf_default(pdf_file: str, top_k: int = 20):
    result_dict = load_result_dict()
    if pdf_file in result_dict:
        try:
            # load that file path with json
            result = json.load(
                open(result_dict[pdf_file], 'r'))
            print(f"Loaded from cache {pdf_file}")
            return result
        except:
            result_dict.pop(pdf_file)

    # create a new db with random unique name
    result_path = f"{PDF_RESULT_DIR}/" + str(uuid.uuid4())
    result_dict[pdf_file] = result_path
    while result_path in result_dict.values():
        result_path = f'{PDF_RESULT_DIR}/' + str(uuid.uuid4())
    # create the db
    llm = get_gpt4_llm()
    index = create_or_get_pdf_db(pdf_file)
    retriever = index.vectorstore.as_retriever()
    retriever.search_kwargs = {"k": top_k}
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           chain_type="stuff",
                                           retriever=retriever)

    paper_queries = get_default_paper_query()
    joined_query = "\n".join([
        f"{i}. {q}"
        for i, q in zip(range(1,
                              len(paper_queries) + 1), paper_queries)
    ])
    result = qa_chain(joined_query)
    with open(result_path, 'w') as f:
        json.dump(result, f)
    save_result_dict(result_dict)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_files', nargs='+', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--light', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    for pdf_file in args.pdf_files:
        pdf_file = f"{OBSIDIAN_PATH}/{pdf_file}"
        if args.light:
            result = query_pdf_summarize(pdf_file)
        else:
            result = query_pdf_summarize_default(pdf_file)
        print(f"Result for {pdf_file}: ", result)
