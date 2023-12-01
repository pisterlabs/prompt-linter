#!/usr/bin/env python3

import chromadb
import concurrent.futures
import fnmatch
import hashlib
import os
import re
import requests
import sys
import time
from bs4 import BeautifulSoup
from chromadb.config import Settings
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AskMyFiles:
    def __init__(self, filename=None, using_stdin=False):
        self.filename = filename
        self.db_folder = '.vectordatadb'
        self.db_path = os.path.join(os.getcwd(), self.db_folder)
        self.relative_working_path = self.db_path + "/../"
        if filename is None:
            self.working_path = os.getcwd()
            self.recurse = True
        else:
            if os.path.isdir(filename):
                self.working_path = os.path.abspath(filename)
                self.recurse = True
            else:
                self.working_path = os.path.dirname(os.path.abspath(filename))
                self.recurse = False

        self.askhints_file = ".askmyfileshints"
        self.askignore_file = ".askignore"
        self.askhints_path = f"{self.relative_working_path}{self.askhints_file}"
        self.collection_name = "filedata"
        self.chromadb = None
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=self.api_key)

        self.max_excerpt_chars = 25000
        self.openai_model = "gpt-3.5-turbo-16k"
        self.model_temperature = 0.6
        self.chunk_size = 1000
        self.chunk_overlap = 100 
        self.using_stdin = using_stdin

    def load_db(self):
        if self.chromadb is None:
            self.chromadb = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.db_path))
            self.files_collection = self.chromadb.get_or_create_collection(self.collection_name)
        if self.files_collection is None:
            self.files_collection = self.chromadb.get_or_create_collection(self.collection_name)

    def persist_db(self):
        self.chromadb.persist()

    def reset_db(self):
        self.load_db()
        self.chromadb.reset()

    def file_info(self,filename):
        self.load_db()
        file_hash = hashlib.sha256(filename.encode()).hexdigest()
        print(f"Finding '{filename}' ({file_hash})...")
        found_files = self.files_collection.get(where={"source": filename})
        print(found_files)


    def join_strings(self,lst):
        result = ''
        for item in lst:
            if isinstance(item, list):
                result += self.join_strings(item) + '\n\n\n'
            else:
                result += item + '\n\n\n'
        return result.strip()

    def process_query_result(self, documents):
        output = []
        max_excerpt_chars = self.max_excerpt_chars
        doc_count = len(documents['metadatas'][0])
        references = [documents['metadatas'][0][index]['source'] for index in range(doc_count - 1)]
        for index in range(0, doc_count - 1):
            output.append(f"""### Start Excerpt from file source {documents['metadatas'][0][index]['source']}
{documents['documents'][0][index]}
### End Excerpt from file source {documents['metadatas'][0][index]['source']}""")

        return [references, self.join_strings(output)[:max_excerpt_chars]]

    def query_db(self, string ):
        max_results = 50
        self.load_db()
        query_embedding = self.embeddings_model.embed_query(string)
        result = self.files_collection.query(query_embeddings=[query_embedding],n_results=max_results,include=['documents','metadatas'])
        return self.process_query_result(result)

    def list_files(self):
        self.load_db()
        results = self.files_collection.get(
            where={"source": { "$ne": "FILELISTQUERYDUMMYCOMPARISON"}},
            include=["metadatas"]
        )

        files = sorted(set([results['metadatas'][index]['source'] for index in range(len(results['metadatas']) - 1)]))

        print("\n".join(files))

        return True


    def get_ignore_list(self):
        ignore_files = []

        ignore_files.append(self.db_folder)
        ignore_files.append('.git')
        image_formats = [ 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'ico', 'webp', 'svg', 'eps', 'raw', 'cr2', 'nef', 'orf', 'sr2', 'heif', 'bat', 'jpe', 'jfif', 'jif', 'jfi' ]
        for ext in image_formats:
            ignore_files.append(f"/*.{ext}")

        askignore_path = os.path.join(self.relative_working_path, self.askignore_file)
        if os.path.exists(askignore_path):
            with open(askignore_path, "r") as file:
                for line in file.read().splitlines():
                    ignore_files.append(line.strip())

        return ignore_files

    def get_file_list(self):
        if not self.recurse:
            relative_file_path = os.path.relpath(self.filename, self.relative_working_path)
            return [relative_file_path]

        ignore_files = self.get_ignore_list()
        use_ignore = len(ignore_files) > 0

        file_list = []
        for root, dirs, files in os.walk(self.working_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_file_path = os.path.relpath(file_path, self.relative_working_path)

                if not use_ignore:
                    file_list.append(relative_file_path)
                    continue

                if not any(pattern == file_path or pattern in file_path or fnmatch.fnmatch(file_path, pattern) for pattern in ignore_files):
                    file_list.append(relative_file_path)

        return file_list

    def remove_file(self,file_name):
        self.load_db()
        file_list = []
        if os.path.isdir(file_name):
            print(f"Removing all files in {file_name} from database...")
            for root, dirs, files in os.walk(file_name):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_file_path = os.path.relpath(file_path, self.relative_working_path)
                    file_list.append(relative_file_path)
        else:
            file_list = [file_name]

        found_ids = []
        files_for_deletion = []
        for file_path in file_list:
            found_file = self.files_collection.get(where={"source": file_path},include=['metadatas'])
            found_count = len(found_file['ids'])
            if found_count > 0:
                found_ids += found_file['ids']
                files_for_deletion += [found_file['metadatas'][index]['source'] for index in range(found_count) ]

        found_ids = list(set(found_ids))
        files_for_deletion = list(set(files_for_deletion))

        if found_ids == []:
            print("File not found in database.")
            return

        print("Removing the following files from the database:")
        print(" - " + "\n - ".join(files_for_deletion))
        self.files_collection.delete(ids=found_ids)
        self.persist_db()

        return True

    def vectorize_text(self, text):
        return self.embeddings_model.embed_query(text)

    def vectorize_chunk(self, chunk, metadata, index):
        embedding = self.vectorize_text(chunk)
        cid = f"{metadata['file_hash']}-{index}"
        return {"id": cid, "document": chunk, "embedding": embedding, "metadata": metadata}

    def vectorize_chunks(self, chunks, metadata):
        max_threads = min(len(chunks), 5)
        vectorized_chunks = {}
        cindex = 1
        iterator = iter(chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            for chunk_group in zip(*[iterator] * max_threads):
                starting_index = cindex
                num_threads = min(max_threads, len(chunk_group))
                futures = []
                for thread_index in range(num_threads):
                    futures.append(executor.submit(self.vectorize_chunk, chunk_group[thread_index], metadata, cindex))
                    cindex += 1
                i = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    chunk_index = starting_index + i
                    vectorized_chunks[f"chunk-{chunk_index}"] = result
                    print(".",end="",flush=True)
                    i += 1
                concurrent.futures.wait(futures)

        return vectorized_chunks

    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            try:
                if os.path.splitext(file_path)[1] == '.pdf':
                    # PDF Processing
                    loader = PyPDFLoader(file_path)
                    pages = loader.load_and_split()
                    content = []
                    for page in pages:
                        content.append(str(page.page_content))
                    return self.join_strings(content)
                else:
                    # Plain Text Processing
                    return file.read()
            except Exception as e:
                print(f"Error reading {file_path}...[Skipped]")
                print
                return None

    def save_vectorized_chunks(self, vectorized_chunks, group_size=10):
        chunk_keys = list(vectorized_chunks.keys())
        if len(chunk_keys) == 0:
            return False

        batches = [chunk_keys[i:i+group_size] for i in range(0, len(chunk_keys), group_size)]

        chunk_keys = list(vectorized_chunks.keys())
        for batch in batches:
            self.files_collection.add(
                ids=[vectorized_chunks[cid]['id'] for cid in batch],
                embeddings=[vectorized_chunks[cid]['embedding'] for cid in batch],
                documents=[vectorized_chunks[cid]['document'] for cid in batch],
                metadatas=[vectorized_chunks[cid]['metadata'] for cid in batch]
            )
            print("+", end='', flush=True)
        self.persist_db()

        return True

    def split_text(self, content):
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return splitter.split_text(content)

    def add_webpage(self, url):
        start_time = time.time()
        self.load_db()

        metadata = {
            "source": url,
            "file_path": url,
            "file_modified": time.time(),
            "file_hash": hashlib.sha256(url.encode()).hexdigest()
        }

        print(f"Fetching '{url}'...",end='',flush=True)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text()
        else:
            print(f"[Failed: {response.status_code}]")
            return False

        print(f"Creating embeddings...",end='',flush=True)
        chunks = self.split_text(content)
        chunk_count = len(chunks)
        print(f"[{len(chunks)} chunks]",end='',flush=True)
        vectorized_chunks = self.vectorize_chunks(chunks, metadata)
        self.files_collection.delete(where={"file_hash": metadata["file_hash"]})
        self.save_vectorized_chunks(vectorized_chunks)

        elapsed_time = max(1, int( time.time() - start_time ))
        print(f"[OK] [{elapsed_time}s]", flush=True)

    def process_file(self,file_path):
        start_time = time.time()
        self.load_db()

        # Get file meta information
        metadata = {
            "source": file_path,
            "file_path": file_path,
            "file_modified": os.path.getmtime(file_path),
            "file_hash": hashlib.sha256(file_path.encode()).hexdigest()
        }

        # File exists?
        existing_record = self.files_collection.get(where={"file_hash": metadata["file_hash"]})
        existing = len(existing_record['ids']) != 0 and len(existing_record['metadatas']) != 0
        if existing:
            file_updated = existing_record['metadatas'][0]["file_modified"] < metadata["file_modified"]
        else:
            file_updated = True

        # Skip File?
        skip_file = existing and not file_updated
        if skip_file:
            return False

        print(f"Creating File Embeddings for: {file_path}...",end='',flush=True)

        # Read content and split
        content = self.read_file(file_path)
        if len(content) < 10 and content.strip() == '':
            print(f"[EMPTY]", flush=True)
            return False

        chunks = self.split_text(content)
        print(f"[{len(chunks)} chunks]",end='',flush=True)

        # Vectorize Chunks
        vectorized_chunks = self.vectorize_chunks(chunks, metadata)
        self.files_collection.delete(where={"file_hash": metadata["file_hash"]})
        self.save_vectorized_chunks(vectorized_chunks)

        # Print status
        elapsed_time = max(1, int( time.time() - start_time ))
        print(f"[OK] [{elapsed_time}s]", flush=True)

        return True

    def load_files(self):
        print("Updating AskMyFiles database...")
        saved_files = False
        for file_path in self.get_file_list():
            try:
                file_saved = self.process_file(file_path)
            except:
                print("Processing Error!")
                file_saved = False
            saved_files = file_saved or saved_files

        return saved_files

    def get_hints(self):
        if os.path.exists(self.askhints_path):
            with open(self.askhints_path, "r") as file:
                return file.read()
        else:
            return ''

    def ask(self, query):
        llm = ChatOpenAI(temperature=self.model_temperature,model=self.openai_model)

        # First Pass
        template = """
        [
        Important Knowledge from MyAskmyfilesLibrary:
        BEGIN Important Knowledge
        {excerpts}
        END Important Knowledge
        ]

        [
        {hints}
        ]

        [
        Start with and prioritize knowledge from MyAskmyfilesLibrary when you answer my question.
        Answer in a very detailed manner when possible.
        If the question is regarding code: prefer to answer using service objects and other abstractions already defined in MyAskmyfilesLibrary and follow similar coding conventions.
        If the question is regarding code: identify if there is a tags file present to inform your answers about modules, classes, and methods.
        ]

        ### Question: {text}
        ### Answer:
        """

        prompt_template = PromptTemplate(input_variables=["text","excerpts","hints"], template=template)
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)
        if not self.using_stdin:
            print("...THINKING...", end='', flush=True)
        local_query_result = self.query_db(query)
        first_answer = answer_chain.run(excerpts=local_query_result[1],hints=self.get_hints(),text=query)

        # Second Pass
        index = first_answer.find("Sources:")
        sources = ""
        if index != -1:
            sources = text[index + len("Sources:"):]

        second_pass_query = f"""
        [
        Consider the following first question and answer:
        Question: {query}
        Answer: {first_answer}

        Sources: {sources}
        ]

        Reconsider the first Question and Answer to answer the following question:
        {query}
        """

        if not self.using_stdin:
            print("THINKING MORE...", end='', flush=True)
        local_query_result2 = self.query_db(second_pass_query)
        second_answer = answer_chain.run(excerpts=local_query_result2[1],hints=self.get_hints(),text=second_pass_query)

        # Output
        if not self.using_stdin:
            print("\n=====================================================")
        print(second_answer)
        if not self.using_stdin:
            print("\n\n=Sources=")
            print(" ".join(list(set(local_query_result[0]))))

if __name__ == "__main__":
    if not sys.stdin.isatty():
        query = "\n".join(sys.stdin.readlines())
        service = AskMyFiles(using_stdin=True)
        service.ask(query)
        sys.exit()

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "ask":
            query = sys.argv[2]
            service = AskMyFiles()
            service.ask(query)
            sys.exit()

        if command == "add":
            path = sys.argv[2]
            if path.startswith('http'):
                service = AskMyFiles()
                service.add_webpage(path)
                sys.exit()

            service = AskMyFiles(path)
            service.load_files()
            sys.exit()

        if command == "remove":
            path = sys.argv[2]
            service = AskMyFiles()
            service.remove_file(path)
            sys.exit()

        if command == "info":
            path = sys.argv[2]
            service = AskMyFiles()
            service.file_info(path)
            sys.exit()

        if command == "add_webpage":
            url = sys.argv[2]
            service = AskMyFiles()
            service.add_webpage(url)
            sys.exit()

        if command == "list":
            service = AskMyFiles()
            service.list_files()
            sys.exit()


        service = AskMyFiles()
        query = ''.join(sys.argv[1:])
        service.ask(query)
        sys.exit()

    else:
        print("askmyfiles ask 'question' or askmyfiles add 'path/dir'")
