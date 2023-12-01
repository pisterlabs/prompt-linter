import os
import pickle
import colorama
import warnings
from dotenv import load_dotenv
from filetype import guess
from langchain.document_loaders import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load the .env file
load_dotenv() 

"""
Suppress UserWarnings for deprecated apply_and_parse method,
Rectify this in the future as and when the library is updated.
/Users/amart/Projects/pdf_ChAi/chAi_venv/lib/python3.9/
site-packages/langchain/chains/llm.py:303: 
UserWarning: The apply_and_parse method is deprecated, 
instead pass an output parser directly to LLMChain.
warnings.warn(
"""
warnings.simplefilter(action='ignore', category=UserWarning)


class PDF_AI:
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.file_path = self.user_filepath_input()
        self.file_name = self.file_path.replace("/", "+")
        self.cache_directory = "cache"
        self._file_type = None
        self._file_content = None
        self._file_splitter = None
        self._embeddings = None
        self._document_search = None
        self._chain = None

    @property
    def file_type(self):
        if not self._file_type:
            self._file_type = self.detect_document_type()
        return self._file_type
    
    @property
    def file_content(self):
        if not self._file_content:
            self._file_content = self.extract_file_content()
        return self._file_content
    
    @property
    def file_splitter(self):
        if not self._file_splitter:
            try:
                self._file_splitter = self.load_chunks_from_file("chunks_cache.pkl")
            except (FileNotFoundError, pickle.UnpicklingError):
                self._file_splitter = self.create_chunks().split_text(self.file_content)
                self.save_chunks_to_file(self._file_splitter, "chunks_cache.pkl")
        return self._file_splitter
    
    @property
    def embeddings(self):
        if not self._embeddings:
            self._embeddings = self.create_embeddings()
        return self._embeddings

    @property
    def document_search(self):
        if not self._document_search:
            self._document_search = self.get_doc_search(self.file_splitter, self.embeddings)
        return self._document_search

    @property
    def chain(self):
        if not self._chain:
            self._chain = self.create_chain()
        return self._chain
   
    def detect_document_type(self):
        guess_file = guess(self.file_path)
        file_type = ""
        image_types = ['jpg', 'jpeg', 'png', 'gif']
        if(guess_file.extension.lower() == "pdf"):
            file_type = "pdf"
        elif(guess_file.extension.lower() in image_types):
            file_type = "image"
        else:
            file_type = "unknown"
        return file_type
    
    def extract_file_content(self):
        if(self.file_type == "pdf"):
            loader = UnstructuredFileLoader(self.file_path)
        elif(self.file_type == "image"):
            loader = UnstructuredImageLoader(self.file_path)
        documents = loader.load()
        documents_content = '\n'.join(doc.page_content for doc in documents)
        return documents_content
    
    def create_chunks(self):
        text_splitter = CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 1500,
            chunk_overlap  = 200,
            length_function = len,
        )
        return text_splitter
    
    def create_embeddings(self):
        embeddings = OpenAIEmbeddings()
        return embeddings
    
    def get_doc_search(self, file_chunks, embeddings):
        return FAISS.from_texts(file_chunks, embeddings)
    
    def create_chain(self):
        return load_qa_chain(OpenAI(), chain_type = "map_rerank",  
                      return_intermediate_steps=True)
    
    def save_chunks_to_file(self, chunks, filename):
        with open(filename, 'wb') as file:
            pickle.dump(chunks, file)

    def load_chunks_from_file(self, filename):
        if not os.path.exists(self.cache_directory):
            os.mkdir(self.cache_directory)
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def create_or_load_cache(self):
        # Attempt to load cached chunks
        try:
            file_splitter = self.load_chunks_from_file(f"cache/chunks_cache+{self.file_name}.pkl")
        except (FileNotFoundError, pickle.UnpicklingError):
            print(colorama.Fore.LIGHTMAGENTA_EX + "No cached chunks found." + colorama.Fore.RESET)
            print(colorama.Fore.LIGHTMAGENTA_EX + "Creating chunks..." + colorama.Fore.RESET)
            file_content = self.extract_file_content()
            file_splitter = self.create_chunks().split_text(file_content)
            # Save the chunks to a file for future use
            self.save_chunks_to_file(file_splitter, f"cache/chunks_cache+{self.file_name}.pkl")
            print(colorama.Fore.LIGHTMAGENTA_EX + "Chunks created & cached." + colorama.Fore.RESET)
        return file_splitter
    
    def chat_with_file(self, query):
        file_splitter =  self.create_or_load_cache()
        embeddings = self.create_embeddings()
        document_search = self.get_doc_search(file_splitter, embeddings)
        documents = document_search.similarity_search(query)
        results = self.chain({
                            "input_documents": documents,
                            "question": query
                        }, 
                        return_only_outputs=True)
        #print(results)
        results = results['output_text']
        return results
    
    def loop_chat_with_file(self):
        if self.create_or_load_cache():
            while True:
                query = input(colorama.Fore.BLUE + "Q: " + colorama.Fore.RESET)
                if query == "exit" or query == '':
                    break
                results = self.chat_with_file(query)
                answer = results
                print(colorama.Fore.GREEN + f"R: {answer}" + colorama.Fore.RESET)
                print("------------------------------------------------------------------------------")

    def user_filepath_input(self):
        file_path = input("Enter the file path: ")
        self.filepath = file_path
        return file_path