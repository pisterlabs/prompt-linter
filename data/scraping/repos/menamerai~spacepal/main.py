import re
import torch
import argparse

import numpy as np
import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader
from pathlib import Path
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.llms.cohere import Cohere
from langchain.chains import RetrievalQA

from transformers import pipeline

from typing import Optional, List, Dict, Tuple, Any, Union

import pickle
import os

Model = Dict[str, Union[Any,  RetrievalQA]]

model_name = "teknium/Phi-Hermes-1.3B"
nli_model = "sileod/deberta-v3-base-tasksource-nli"
# model_name = "teknium/Puffin-Phi-v2"
device = 0 if torch.cuda.is_available() else -1

spec_toc_pattern = "[0-9]+\.[0-9\.]*\s?[a-z A-Z0-9\-\,\(\)\n\?]+\s?[\.\s][\.\s]+\s?[0-9]+"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="teknium/Phi-Hermes-1.3B")
    parser.add_argument("-k", "--key", type=str)
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    if args.model_name == "cohere" and args.key is None:
        raise ValueError("must provide API key if using cohere backend")
    return args

def get_toc_pages(pages: List) -> List[Any]:
    """
    Extract table of content information, chiefly section title and corresponding page
    Used in content classification of prompt to ground model
    --------------------------------------------------------
    pages: List

    returns
    --------------------------------------------------------
    Table of content pages in a list
    """
    state = 0 # 0 means we are looking for TOC start, 1 means we're parsing regex
    i = 0
    toc_pages = []
    while i < len(pages):
        page = pages[i]
        text = page.extract_text()
        match state:
            case 0:
                if "table of contents" in text.lower():
                    state = 1
                else:
                    i += 1
            case 1:
                matches = re.findall(spec_toc_pattern, text)
                if len(matches) > 0:
                    toc_pages += [page]
                else:
                    break
                i += 1

    max_pg_number = 0
    for i, page in enumerate(toc_pages):
        text = page.extract_text()
        matches = [x for x in re.findall(spec_toc_pattern, text)]
        pg_number_pattern = "[0-9]+$"
        pg_number_strs = [re.findall(pg_number_pattern, match.strip()) for match in matches]
        pg_numbers = [int(num[0]) for num in pg_number_strs if len(num) > 0] + [0]
        if max(pg_numbers) < max_pg_number:
            return toc_pages[:i] 
        else:
            max_pg_number = max(pg_numbers)

    return toc_pages

def get_spec_entry(pages: List, title: str, entries: Dict[str, Dict]) -> Tuple[int]:
    """
    For a specific specification document, get the text corresponding to that entry
    --------------------------------------------------------
    pages: list

    title: str

    entries: Dict[str, Dict]

    returns
    --------------------------------------------------------
    An entry-specific text
    """
    entry = entries[title]
    pg_num = entry['page_number'] - 1
    spec_num, spec_title = entry['spec_number'], entry['spec_title']
    for page in pages[pg_num:]:
        content = page.extract_text()
        content_lines = [line.strip() for line in content.split("\n") if line.strip() != ""]

        state = 0
        spec_content = ""
        for line in content_lines:
            match state:
                case 0:
                    if spec_num in line and spec_title in line:
                        spec_content += (line + "\n")
                        state = 1
                case 1:
                    found_titles = [t for t, v in entries.items() if v['spec_number'] in line and v['spec_title'] in line and t != title]
                    if len(found_titles) == 0:
                        spec_content += (line + "\n")                    
                    else:
                        break
        if spec_content != "":
            return spec_content
    return None

def get_subspec_entries(title: str, entries: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    For a sub-specific specification document, get the text corresponding to that entry
    --------------------------------------------------------
    title: str

    entries: Dict[str, Dict]

    returns
    --------------------------------------------------------
    An entry-specific text
    """
    spec = entries[title]['spec']
    if spec is None:
        return {}
    page_num = entries[title]['page_number']
    spec_num = entries[title]['spec_number']
    level = spec_num.count(".")

    subspec_idxs = []
    lines = spec.split("\n")
    for i, line in enumerate(lines[1:]):
        if spec_num in line:
            subspec_idxs += [i + 1]

    subspec_entries = {}
    for idx in subspec_idxs:
        subspec_tokens = lines[idx].split()
        subspec_num, subspec_title = subspec_tokens[0], ' '.join(subspec_tokens[1:])
        if subspec_num.count(".") > level + 1:
            continue
        subspec_entries[lines[idx]] = {
            "page_number": page_num,
            "spec_number": subspec_num,
            "spec_title": subspec_title
        }
    return subspec_entries

def get_spec_list(pdf_path: str, verbose: bool = False) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Get the list of the specifcations from the document based on the table of content
    --------------------------------------------------------
    pdf_path: str

    verbose: bool = False

    returns
    --------------------------------------------------------
    A tuple of list of specifcations and dictionary of entries
    """
    pdf_reader = PdfReader(pdf_path)
    entries = get_toc_entries(pdf_reader.pages)

    specs = []
    entry_iter = entries.keys()
    if verbose:
        entry_iter = tqdm(entry_iter, total=len(entries))
    for title in entry_iter:
        spec_entry = get_spec_entry(pdf_reader.pages, title, entries)
        if spec_entry is None:
            entries[title]['spec'] = ""
        else:
            specs += [spec_entry]
            entries[title]['spec'] = spec_entry

    tmp = {}
    for k in entries.keys():
        tmp.update(get_subspec_entries(k, entries))

    copy = entries.copy()
    copy.update(tmp)
    for k in tmp.keys():
        spec_entry = get_spec_entry(pdf_reader.pages, k, copy)
        if spec_entry is None:
            tmp[k]['spec'] = ""
        else:
            tmp[k]['spec'] = spec_entry

    entries.update(tmp)
    lens = [len(s) for s in specs]
    page_one_chunks = []
    if len(lens) != 0:
        page_one_content = pdf_reader.pages[0].extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=int(np.mean(lens)),
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        page_one_chunks = text_splitter.split_text(page_one_content)
    return page_one_chunks + specs, entries

def get_toc_entries(pages: List) -> Dict[str, Dict]:
    """
    Get all the entries in table of content
    --------------------------------------------------------
    pages: list

    --------------------------------------------------------
    A dictionary of all the entries
    """
    pages = get_toc_pages(pages)
    split_pattern = "\.\.+" # match 2 or more dots
    entries = {}
    for page in pages:
        text = page.extract_text()
        matches = [x for x in re.findall(spec_toc_pattern, text)]
        for match in matches:
            match_components = re.split(split_pattern, match)
            match_components[0] = match_components[0].replace("\n", "").strip()
            match_components[-1] = match_components[-1].split()[0].replace(".", "").strip()
            tokens = match_components[0].split() 
            spec_num, spec_title = tokens[0], ' '.join(tokens[1:])
            try:
                pg_num = int(match_components[-1])
            except Exception as e:
                continue
            entries[match_components[0]] = {
                "page_number": pg_num,
                "spec_number": spec_num,
                "spec_title": spec_title
            }
    return entries

def load_pdf_to_chunks(pdf_path: str, chunk_size: Optional[int] = None, verbose: bool = False) -> List[str]:
    """
    Load the pdf into chunks to be processed by the tokenizer
    --------------------------------------------------------
    pdf_path: str

    chunk_size

    verbose: bool

    returns
    --------------------------------------------------------
    A list of text chunks
    """
    specs, toc_entries = get_spec_list(pdf_path, verbose=verbose) 
    if chunk_size is not None:
        lens = [len(s) for s in specs]
        idx = [i for i, l in enumerate(lens) if l > chunk_size]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=40,
            length_function=len
        )
        for i in idx:
            specs += text_splitter.split_text(specs[i])
    return specs, toc_entries

def binary_to_pdf(bin, dir: str, name: str):
    """
    Write the binary inputted into PDF file on folder
    --------------------------------------------------------
    bin

    dir: str

    name: str
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(Path(dir) / name, "wb") as pdf_out:
        pdf_out.write(bin)

@st.cache_resource
def init_chain(model_name: str, pdf_path: str, key: Optional[str] = None) -> Tuple[Model, Dict[str, Dict]]:
    """
    Initialize langchain and huggingface backend given model name/api key
    --------------------------------------------------------
    model_name: str

    pdf_path: str

    key: Optional[str] = None

    returns
    --------------------------------------------------------
    A tuple of the model and ToC dict
    """
    chunks, toc_entries = load_pdf_to_chunks(pdf_path, verbose=True)
    store_name = pdf_path[:-4]
        
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
        #st.write("Already, Embeddings loaded from the your folder (disks)")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        #Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(vectorstore,f)

    retriever = vectorstore.as_retriever()
    
    if model_name == "cohere":
        llm = Cohere(cohere_api_key=key)
        prompt_template = "You are a superintelligent AI assistant that excels in handling technical documents. Use the following context to answer the question, and cite specification numbers for context used. Do not make up information, use the document for all of your thinking. The document is grammatically correct. Take a deep breath, be as specific as possible, failure is not an option. If you don't know something, just say it.\nSpecification Context:\n{context}{question}. Be as specific as possible and cite all used sources.\nAnswer:"
    else:
        prompt_template = "### Instruction: You are an AI assistant NASA missions used specifically to proof-read their documentations. Use the following context to answer the question. Base your answers on the context given, do not make up information.\nContext:\n{context}\nQuestion: {question}\n### Response: \n"
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=device,  # -1 for CPU
            batch_size=1,  # adjust as needed based on GPU map and model size.
            model_kwargs={"do_sample": True, "temperature": 0.8, "max_length": 2048, "torch_dtype": torch.bfloat16, "trust_remote_code": True},
        )

    template = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    chain_type_kwargs = {"prompt": template}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    classifier = pipeline("zero-shot-classification", model=nli_model, device=device)
    return {"chain": chain, "classifier": classifier}, toc_entries

def generate_response(question: str, model: Model, toc_entries: Dict[str, Dict] = {}):
    """
    Given pre-trained model object, generate response from model
    --------------------------------------------------------
    question: str

    model: Model

    toc_entires: Dict[str, Dict]
    """
    context = ""
    # if we're using ToC to augment, hack extra sources into RAG through the question string
    if len(toc_entries) > 0:
        toc_labels = list(toc_entries.keys())
        results = model['classifier'](question, toc_labels)
        score_idxs = np.argsort(results['scores'])[::-1][:25]

        levels = [toc_entries[results['labels'][i]]['spec_number'].count('.') for i in score_idxs]
        level_idxs = np.argsort(levels)[::-1][:3]

        for i in level_idxs:
            label = results['labels'][score_idxs[i]]
            context += (toc_entries[label]['spec'] + "\n")

    question = f"{context}\nQuestion: {question}"
    response = model['chain'].run(question)
    return response

if __name__ == "__main__":
    args = parse_args()
    model, toc_entries = init_chain(args.model_name, args.path, key=args.key)
    question = input("Input Question: ")
    response = generate_response(question, model, toc_entries)
    print(response)