import os
import json
import argparse
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter
import gspread
from google.auth import default
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
import openai
import gspread
import nltk
# nltk.download('punkt')
from google.oauth2.service_account import Credentials


class DataExtractor:

    def __init__(self, dir_path):
        self.dir_path = dir_path


    def extract_data(self):
        with open(self.filename, 'r') as f:
            data = f.read()
        return data
    

    def load_pdfs(self):

        dir_loader = DirectoryLoader(self.dir_path, glob="**/*.pdf", 
                                loader_cls=PyPDFLoader, 
                                show_progress=True,
                                use_multithreading=True,
                                silent_errors=True)
        docs = dir_loader.load()
        print(f"\nNumber of docs after initial loading: {len(docs)}, from: {self.dir_path}")

        return docs
    

    def chunk_docs(self, docs, chunk_size=2000, nltk=True, spacy=False, recursive=False):

        if nltk:
            text_splitter = NLTKTextSplitter(chunk_size=chunk_size,
                                            chunk_overlap=0)
        elif spacy:
            text_splitter = SpacyTextSplitter(chunk_size=chunk_size)
        elif recursive:
            text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                    length_function=len,
                    separators=["\n\n\n","\n\n", "\n", ". ", " ", ""],)
        else:
            text_splitter = CharacterTextSplitter(
                                    separator='\n',
                                    chunk_size=chunk_size,
                                    chunk_overlap=0,
                                    length_function=len,)

        all_text = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        chunks = text_splitter.create_documents(all_text, metadatas=metadatas)
        print(f"Number of chunks: {len(chunks)}")

        return chunks
    

    def update_metadata_from_file(self, docs, file_path=None):

        if file_path is None:
            file_path = os.path.join(os.getcwd(), "guideline_metadata.csv")
        else:
            file_path = file_path

        metadata_df = pd.read_csv(file_path)
        metadata_df = metadata_df.fillna(" ")
        metadata_df.set_index('file_name', inplace=True)
        metadata_dict = metadata_df.to_dict(orient='index')

        for doc in docs:
            file_name = os.path.basename(doc.metadata['source'])
            try:
                merged_dict = metadata_dict[file_name].copy()
                merged_dict['broad_category'] = merged_dict['broad_category'].lower().strip()
                merged_dict['file_name'] = file_name
                merged_dict.update(doc.metadata)
                doc.metadata = merged_dict
            except KeyError:
                print(f"File name {file_name} not found in metadata file. Skipping...")

        print(f"Updated {len(docs)} docs metadata from: {file_path}")
        
        return docs 


    ### NOT WORKING YET
    ## Problem with authentication
    def update_metadata_from_gsheets(self, docs, key=None):
        
        # Get google sheets key from env
        load_dotenv()
        key = os.environ.get('GOOGLE_SHEETS_KEY')
        credentials = os.environ.get('GOOGLE_SERVICE_ACCOUNT')

        # Google Sheets authorization.
        creds, _ = default()
        creds = Credentials.from_service_account_file(credentials)
        gc = gspread.authorize(creds)

        # Open the Google Spreadsheet by its title.
        workbook = gc.open_by_key(key)
        worksheet = workbook.sheet1

        # Get all values in the sheet and convert it into DataFrame.
        values = worksheet.get_all_values()
        metadata_df = pd.DataFrame(values[1:], columns=values[0])
        metadata_df.set_index('file_name', inplace=True)

        # Convert DataFrame to dictionary.
        metadata_dict = metadata_df.to_dict(orient='index')

        for doc in docs:
            file_name = os.path.basename(doc.metadata['source'])
            merged_dict = metadata_dict[file_name].copy()
            merged_dict['file_name'] = file_name
            merged_dict.update(doc.metadata)
            doc.metadata = merged_dict
        
        print(f"Updated {len(docs)} docs metadata from: {workbook}")

        return docs
    

    def add_chunk_index(self, chunks):

        sources = []
        for chunk in chunks:
            sources.append(chunk.metadata['source'])
        list(set(sources))
        for source in sources:
            chunk_index = 0
            for chunk in chunks:
                if source == chunk.metadata['source']:
                    chunk.metadata['chunk_index'] = chunk_index
                    chunk_index += 1
                else:
                    continue
            total_chunks = chunk_index
            for chunk in chunks:
                if source == chunk.metadata['source']:
                    chunk.metadata['last_chunk_index'] = total_chunks - 1

        print(f"Added chunk_index to metadata of {len(chunks)} chunks")

        return chunks 


    # Create chroma vectorstore
    def get_chroma_vectorstore(self, chunks, 
                                use_openai=True, 
                                persist_directory="chroma_db"):
        # load api keys
        load_dotenv()
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        if use_openai:
            model_name = 'text-embedding-ada-002'
            embeddings = OpenAIEmbeddings(model=model_name,
                                        openai_api_key=openai.api_key)
        else: 
            model_name = "hkunlp/instructor-xl" 
            embed_instruction = "Represent the text from the clinical guidelines"
            query_instruction = "Query the most relevant text from clinical guidelines"
            embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                    embed_instruction=embed_instruction, 
                                                    query_instruction=query_instruction)

        # Create vectorstore
        vectorstore = Chroma.from_documents(documents=chunks,
                                            embedding=embeddings,
                                            persist_directory=persist_directory)

        print(f"Created chroma vectorscore, called: {persist_directory}")

        return vectorstore


    def __call__(self, persist_directory="chroma_db"):
     
        docs = self.load_pdfs()
        chunks = self.chunk_docs(docs, chunk_size=3000, nltk=True)
        chunks = self.update_metadata_from_file(chunks, file_path=None)
        chunks = self.add_chunk_index(chunks)
        vectorstore = self.get_chroma_vectorstore(chunks, use_openai=True,
                                                        persist_directory=persist_directory)

        return vectorstore
    

def main(dir_path=None):

    if dir_path is None:
        # take shell arguments for directory path
        parser = argparse.ArgumentParser()
        parser.add_argument("--dir_path", type=str, default=None)
        args = parser.parse_args()
        dir_path = args.dir_path
    else:
        dir_path = dir_path
        
    extractor = DataExtractor(dir_path)
    persist_directory = os.path.join(os.getcwd(), "vector_db", "chroma_db_ada_embeddings")
    vectorstore = extractor(persist_directory)
  
    return None


if __name__ == "__main__":

    main(dir_path="./guidelines")
 