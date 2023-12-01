import os
import re
import glob
import shutil
import tabula
import requests
import pandas as pd
import streamlit as st
from typing import List
from PyPDF2 import PdfReader
from datetime import datetime
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text


from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


import Config_Paremeters
from Config_Streamlit import *
from dotenv import load_dotenv
load_dotenv()


LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".pdf": (PDFMinerLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".csv": (CSVLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
}


upload_data_directory = Config_Paremeters.UPLOAD_DATA_DIRECTORY
clean_data_directory = Config_Paremeters.CLEAN_DATA_DIRECTORY
temp_directory = Config_Paremeters.TEMP_DIRECTORY
persist_directory = Config_Paremeters.PERSIST_DIRECTORY
cache_directory = Config_Paremeters.CACHE_DIRECTORY
url_file_name = Config_Paremeters.URL_FILE_NAME
uploaded_file_names = Config_Paremeters.UPLOADED_FILE_NAMES
embedding_model = Config_Paremeters.OPENAI_EMBEDDING_MODEL
llm_model = Config_Paremeters.OPENAI_LLM_MODEL
OPENAI_API_KEY = Config_Paremeters.OPENAI_API_KEY

# ==================== Embeddings Functions ====================
embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY)


# ==================== Upload Documents ====================
def upload_documents(uploaded_files):
    num_files_uploaded = 0
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            remove_upload_data_directory()
            if not os.path.exists(upload_data_directory):
                os.makedirs(upload_data_directory)
            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    safe_filename = convert_file_name(uploaded_file)
                    file_path = os.path.join(upload_data_directory, safe_filename)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    num_files_uploaded += 1
    except Exception as e:
        print(f"Error while uploading documents: {e}")
        return False, f"Error while uploading documents: {e}"
    return True, f"Successfully uploaded {num_files_uploaded} files"



def convert_file_name(uploaded_file):
    try:
        base_name, extension = os.path.splitext(uploaded_file.name)
        base_name = re.sub(r'\W+', '_', base_name)[:25]
        safe_filename = base_name + extension
        save_uploaded_file_names(temp_directory, uploaded_file_names, safe_filename)
        return safe_filename
    except Exception as e:
        print(f"Error while converting file name: {e}")
        return False, f"Error while converting file name: {e}"


def save_uploaded_file_names(directory, uploaded_file_names, file_name):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, uploaded_file_names)
        with open(file_path, 'a') as f:
                f.write(file_name + '\n')  
    except Exception as e:
        print(f"Error while saving string to file: {e}")



def replace_source_name(string_1):
    try:
        filename = os.path.join(temp_directory, uploaded_file_names)
        with open(filename, 'r') as f:
            string_list = f.read().splitlines()
        for string_2 in string_list:
            set_1 = set(string_1)
            set_2 = set(string_2)
            common = set_1 & set_2
            ratio = len(common) / len(set_1)
            if ratio >= 0.25:
                return string_2  
        return string_1  
    except Exception as e:
        print(f"Error while replacing source name: {e}")


# ==================== Train Model ====================
def train_model():
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            print("\n")
            print("*" * 100)
            process_start_time = datetime.now()
            print('process_start_time: ', process_start_time)
            print("\n")
            remove_persist_directory()
            clean_documents(upload_data_directory, clean_data_directory)
            loaded_documents = load_documents(clean_data_directory)
            chunks = create_chunks(loaded_documents)
            create_embeddings(chunks)
            process_end_time = datetime.now()
            process_run_time = process_end_time - process_start_time
            run_time_seconds = round(process_run_time.total_seconds())
            print('process_start_time: ', process_start_time, '| process_end_time: ', process_end_time)
            print('process_run_time: ', run_time_seconds, ' seconds')
            print("*" * 100)
            print("\n")
    except Exception as e:
        print(f"Error while training model: {e}")


def clean_documents(input_directory, output_directory):
    try:
        print('\nStarted cleaning documents ...')
        remove_clean_data_directory()
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for file in glob.glob(os.path.join(input_directory, "*")):
            shutil.copy(file, output_directory)
        convert_pdfs_to_text(output_directory, output_directory)
        clean_text_files(output_directory)
        # extract_tables_from_pdfs(output_directory, output_directory)
        remove_unwanted_files(output_directory)
        print('Cleaning documents complete ... Done')
    except Exception as e:
        print(f"Error while cleaning documents: {e}")
        return False, f"Error while cleaning documents: {e}"
    return True, f"Successfully cleaned documents"



def convert_pdfs_to_text(pdf_directory, txt_directory):
    try:
        convert_pdfs_to_text_pypdf2(pdf_directory, txt_directory)
        convert_pdfs_to_text_pdfminer(pdf_directory, txt_directory)
    except Exception as e:
        print(f"Error while converting PDFs to text: {e}")
        return False, f"Error while converting PDFs to text: {e}"


def convert_pdfs_to_text_pypdf2(pdf_directory, txt_directory):
    try:
        print('\nConverting PDF files to text using pypdf2 ...')
        files = os.listdir(pdf_directory)
        for file_name in files:
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, file_name)
                txt_path = os.path.join(txt_directory, os.path.splitext(file_name)[0] + '_pypdf2' + '.txt')
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    text = ''
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text += page.extract_text()
                with open(txt_path, 'w') as txt_file:
                    txt_file.write(text)
                print(f'Converted {file_name} to text.')
        print('All PDF files converted to text ... Done')
    except Exception as e:
        print(f"Error while converting PDFs to text: {e}")
        return False, f"Error while converting PDFs to text: {e}"
    


def convert_pdfs_to_text_pdfminer(pdf_directory, txt_directory):
    try:
        print('\nConverting PDF files to text using pdfminer ...')
        files = os.listdir(pdf_directory)
        for file_name in files:
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, file_name)
                txt_path = os.path.join(txt_directory, os.path.splitext(file_name)[0] + '_pdfminer' + '.txt')
                text = extract_text(pdf_path)
                with open(txt_path, 'w') as txt_file:
                    txt_file.write(text)
                print(f'Converted {file_name} to text.')
        print('All PDF files converted to text ... Done')
    except Exception as e:
        print(f"Error while converting PDFs to text: {e}")
        return False, f"Error while converting PDFs to text: {e}"
    
    

def clean_text_files(directory):
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                remove_long_strings(file_path)
    except Exception as e:
        print(f"Error cleaning text files: {e}")
        return False, f"Error cleaning text files: {e}"
    


def remove_long_strings(file_path):
    try:
        print(f"Removing long strings from {file_path} ...")
        with open(file_path, 'r') as f:
            s = f.read()
        long_strings = re.findall(r'\S{20,}', s)  # remove string of length > 20
        if not long_strings:
            print("No strings of length > 25 found in {filename}.".format(filename=file_path))
            return
        with open(file_path, 'w') as f:
            for long_string in long_strings:
                s = s.replace(long_string, ' ')
            f.write(s)
    except Exception as e:
        print(f"Error while removing long strings: {e}")


# change this function
def extract_tables_from_pdfs(pdf_directory, csv_directory):
    try:
        print('\nExtracting tables from PDF files ...')
        os.makedirs(csv_directory, exist_ok=True)
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_file_path = os.path.join(pdf_directory, pdf_file)
            tables = tabula.read_pdf(pdf_file_path, pages='all', multiple_tables=True)
            for i, table in enumerate(tables):
                csv_file_path = os.path.join(csv_directory, f"{pdf_file.replace('.pdf', '')}_table_{i}.csv")
                table.to_csv(csv_file_path, index=False)
        print(f"Tables extracted from {len(pdf_files)} PDF files ... Done")
    except Exception as e:
        print(f"Failed to process file {pdf_file_path}: {e}")



def remove_unwanted_files(directory_path):
    try:
        print('\nRemoving unwanted files ...')
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_file_path = os.path.join(directory_path, pdf_file)
            os.remove(pdf_file_path)
        print('All unwanted files removed ... Done')
    except Exception as e:
        print(f"Error while removing unwanted files: {e}")



def load_single_document(file_path: str) -> Document:
    try:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            loaded_list = loader.load()
            if loaded_list is None or len(loaded_list) == 0:
                print(f"Warning: No documents were loaded from file: {file_path}")
                return None
            loaded_doc = loaded_list[0]
            if loaded_doc is None:
                print("Warning: Loaded document is None from file: " + file_path)
                return None
            return loaded_doc
        raise ValueError(f"Unsupported file extension '{ext}' in file: {file_path}")
    except Exception as e:
        print(f"Error while loading single document from file: {file_path}. Error details: {e}")



def load_documents(source_dir: str) -> List[Document]:
    try:
        print(f"\nLoading documents from {source_dir} ... ")
        all_files = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        loaded_documents = []
        for file_path in all_files:
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                document = load_single_document(file_path)
                if document is not None:
                    loaded_documents.append(document)
        print(f"Loaded {len(loaded_documents)} documents ... Done")
        return loaded_documents
    except Exception as e:
        print(f"Error while loading documents: {e}")



def filter_chunks(chunks):
    try:
        filtered_chunks = [element for element in chunks if len(str(element.page_content)) > 250] 
        return filtered_chunks
    except Exception as e:
        print(f"Error while filtering chunks: {e}")



def create_chunks(documents):
    try:
        print(f"\nCreating chunks ... ")
        if not documents:
            print("No new documents to load")
            exit(0)
        chunks = []
        chunk_sizes = [300, 600, 900]
        chunk_overlaps = [60, 120, 180]
        for chunk_size, chunk_overlap in zip(chunk_sizes, chunk_overlaps):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
            this_chunks = text_splitter.split_documents(documents)
            filtered_chunks = filter_chunks(this_chunks)
            print(f"Unfiltered chunks: {len(this_chunks)} | Filtered chunks: {len(filtered_chunks)}")
            chunks.extend(filtered_chunks)
        print(f"Split into {len(chunks)} chunks of text (max. {chunk_size} tokens each)")
        print(f"Creating chunks ... Done")
        return chunks
    except Exception as e:
        print(f"Error while creating chunks: {e}")


def create_embeddings(texts):
    try:
        print(f"\nCreating embeddings ... ")
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
        db.persist()
        db = None
        print(f"Created embeddings ... Done")
        print(f"created vectorstores and persisted to {persist_directory} ... Done")
    except Exception as e:
        print(f"Error while creating embeddings: {e}")




def create_chain():
    try:
        print(f"\nCreating conversation chain ... ")
        stored_vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = stored_vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        llm = ChatOpenAI(model=llm_model, temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        print(f"Created chain ... Done")
        return chain
    except Exception as e:
        print(f"Error while creating conversation chain: {e}")



def get_response(query, chat_history):
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            valid_conter = 0
            selected_page_content_list = []
            chain = create_chain()
            result = chain({"question": query, "chat_history": chat_history})
            question, answer, source_documents = result['question'], result['answer'], result['source_documents']
            print("\n")
            print(result)
            chat_response = f"\nAI Assistant: {answer}\n"
            for counter, document in enumerate(source_documents, start=1):
                selected_page_content = document.page_content[:100]
                if selected_page_content not in selected_page_content_list:
                    valid_conter += 1
                    selected_page_content_list.append(selected_page_content)
                    chat_response += f"\n"
                    chat_response += f"{'-' * 100}"
                    source_name = document.metadata['source']
                    modified_source_name = replace_source_name(source_name)
                    chat_response += f"\n> {modified_source_name} #{valid_conter}:\n"
                    chat_response += document.page_content + '\n'
            return question, answer, chat_response
    except Exception as e:
        print(f"Error while getting response: {e}")
        return None, None, None





# ==================== Reset Model ====================
def reset_model():
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            remove_upload_data_directory()
            remove_clean_data_directory()
            remove_temp_directory()
            remove_persist_directory()
    except Exception as e:
        print(f"Error while resetting model: {e}")






# ==================== Create Text File from URL ====================
def load_data_from_url(base_url):
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            remove_upload_data_directory(upload_data_directory)
            extract_all_urls_from_base_url(base_url, url_file_name)
            create_text_file_from_url(url_file_name, upload_data_directory)
    except Exception as e:
        print(f"Error while loading data: {e}")


def extract_all_urls_from_base_url(base_url):
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        a_tags = soup.find_all('a')
        with open(url_file_name, 'w') as f:
            for tag in a_tags:
                href = tag.get('href')
                if href is not None:  # avoid NoneType objects if any
                    if href.startswith("http"):  
                        f.write(href + '\n')
                    else:
                        f.write(f"{base_url.rstrip('/')}/{href.lstrip('/')}\n")
    except Exception as e:
        print(f"Error while extracting urls from {base_url}: {e}")


def create_text_file_from_url(url_file_name, data_directory):
    try:
        counter = 0
        with open(url_file_name, 'r') as f:
            for url in f.readlines():
                url = url.strip()
                if url:
                    counter += 1
                    output_file = f"document_{counter}.txt"
                    response = requests.get(url)
                    if response.status_code == 200:
                        webpage_content = response.content
                        soup = BeautifulSoup(webpage_content, "html.parser")
                        with open(output_file, "w") as file:
                            paragraphs = soup.find_all('p')
                            for paragraph in paragraphs:
                                file.write(paragraph.get_text())
                                file.write('\n')  # add a newline to separate paragraphs
                        if not os.path.exists(data_directory):
                            os.makedirs(data_directory)
                        shutil.move(output_file, f'{data_directory}/{output_file}')
                        print(f'Text has been written to {output_file}')
                    else:
                        print(f'Failed to retrieve webpage. Status code: {response.status_code}')
    except Exception as e:
        print(f"Error while creating text file from {url_file_name}: {e}")





# ==================== #
# remove directories
def remove_cache_directory():
    try:
        if os.path.exists(cache_directory):
            shutil.rmtree(cache_directory)
    except Exception as e:
        print(f"Error while removing cache directory: {e}")

def remove_upload_data_directory():
    try:
        if os.path.exists(upload_data_directory):
            shutil.rmtree(upload_data_directory)
    except Exception as e:
        print(f"Error while removing data directory: {e}")

def remove_clean_data_directory():
    try:
        if os.path.exists(clean_data_directory):
            shutil.rmtree(clean_data_directory)
    except Exception as e:
        print(f"Error while removing clean data directory: {e}")


def remove_temp_directory():
    try:
        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)
    except Exception as e:
        print(f"Error while removing temp directory: {e}")


def remove_persist_directory():
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
    except Exception as e:
        print(f"Error while removing persist directory: {e}")

def clear_conversation():
    try:
        with st.spinner(f"Please wait, this may take a while..."):
            st.session_state['chat_history'] = []
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['messages'] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
    except Exception as e:
        print(f"Error while clearing conversation: {e}")

