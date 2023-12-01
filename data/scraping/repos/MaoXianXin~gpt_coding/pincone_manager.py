#%%

import pinecone
import openai
from datasets import load_dataset
import time
from tqdm.auto import tqdm
import os
import itertools
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class PineconeManager:
    def __init__(self, api_key, environment, index_name, dimension=None):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.index = None
        self._initialize_pinecone()

    def _initialize_pinecone(self):
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )

    def create_index_if_not_exists(self):
        if self.index_name not in pinecone.list_indexes():
            if self.dimension is None:
                raise ValueError("Dimension must be provided to create a new index.")
            pinecone.create_index(self.index_name, dimension=self.dimension)

    def connect_to_index(self):
        self.index = pinecone.Index(self.index_name)

    def delete_index(self):
        pinecone.delete_index(self.index_name)

    def upsert_vectors(self, data_generator, batch_size=100):
        for ids_vectors_chunk in self._chunks(data_generator, batch_size=batch_size):
            self.index.upsert(vectors=ids_vectors_chunk)

    def query(self, vectors, top_k=5, include_metadata=True):
        res = self.index.query(vectors, top_k=top_k, include_metadata=include_metadata)
        return res

    def _chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))


class OpenAIManager:
    def __init__(self, api_key, api_base, model, chat_model=None):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.chat_model = chat_model
        self._initialize_openai()

    def _initialize_openai(self):
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    def create_embedding(self, input_text):
        res = openai.Embedding.create(model=self.model, input=input_text)
        return res['data'][0]['embedding']

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat_completion(self, messages, max_tokens=2000, temperature=0.1):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response


class FileManager:
    def __init__(self, folder_list, file_exts=None, keyword=None):
        self.folder_list = folder_list
        self.file_exts = file_exts
        self.keyword = keyword

    def get_files_in_folders(self):
        file_list = []  
        for folder_path in self.folder_list:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if (self.file_exts is None or any(file.endswith(ext) for ext in self.file_exts))\
                            and (self.keyword is None or self.keyword in file):
                        file_list.append(os.path.join(root, file))
        return file_list

    def read_files(self):
        text_list = []
        files = self.get_files_in_folders()
        for file_path in files:
            with open(file_path, 'r') as f:
                text_list.append(f.read())
        return text_list


class TextVectorManager:
    def __init__(self, pinecone_manager, openai_manager):
        self.pinecone_manager = pinecone_manager
        self.openai_manager = openai_manager

    def upsert_text_vectors(self, text_list, batch_size=100):
        for i in tqdm(range(0, len(text_list), batch_size)):
            i_end = min(i+batch_size, len(text_list))
            lines_batch = text_list[i: i+batch_size]
            ids_batch = [str(time.time()) for n in range(i, i_end)]
            embeds = [self.openai_manager.create_embedding(line) for line in lines_batch]
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            self.pinecone_manager.upsert_vectors(list(to_upsert))

    def query_text(self, text, top_k=5):
        xq = self.openai_manager.create_embedding(text)
        res = self.pinecone_manager.query([xq], top_k=top_k, include_metadata=True)
        return res

    def query_text_and_get_response(self, query, max_tokens=2000, temperature=0.1, top_k=1):
        res = self.query_text(query, top_k=top_k)

        message_content = (
            "Context:\n"
            + res['matches'][0]['metadata']['text']
            + "\n\n"
            + "Question:\n"
            + query
            + "\n\n"
            + "Answer:\n"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message_content},
        ]

        response = self.openai_manager.chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
        return response
    
    def query_text_and_get_response_poe(self, query, top_k=1):
        res = self.query_text(query, top_k=top_k)

        message_content = (
            "Context:\n"
            + res['matches'][0]['metadata']['text']
            + "\n\n"
            + "Question:\n"
            + query
            + "\n\n"
            + "Answer:\n"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message_content},
        ]

        conversation_string = ""
        for message in messages:
            for key, value in message.items():
                conversation_string += f"{key}:{value}\n"
        print(conversation_string)
        return conversation_string

#%%

if __name__ == "__main__":
    # Example usage:
    API_KEY = "046946b9-2a4f-4b67-a4cb-4f4b79701493"
    ENVIRONMENT = "asia-southeast1-gcp"
    INDEX_NAME = "openai"
    EMBED_DIMENSION = 1536
    OPENAI_API_KEY = "fk210108-nOO4jm9LwK5W4htjDNdnJqyTdgoaoRrw"
    OPENAI_API_BASE = "https://openai.api2d.net/v1"
    OPENAI_MODEL = "text-embedding-ada-002"
    CHAT_MODEL = "gpt-3.5-turbo-16k"

    pinecone_manager = PineconeManager(API_KEY, ENVIRONMENT, INDEX_NAME, EMBED_DIMENSION)
    pinecone_manager.create_index_if_not_exists()
    pinecone_manager.connect_to_index()

    openai_manager = OpenAIManager(OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL, CHAT_MODEL)

    text_vector_manager = TextVectorManager(pinecone_manager, openai_manager)

    #%%

    folder_list = ["/home/mao/github_repo/ChatGLM2-6B",]
    file_exts = ['.py', '.md']
    keyword = None

    file_manager = FileManager(folder_list, file_exts=file_exts, keyword=keyword)
    text_list = file_manager.read_files()

    text_vector_manager.upsert_text_vectors(text_list)

    #%%

    # Perform a query
    query = "what is ChatGLM2-6M model?"
    response = text_vector_manager.query_text_and_get_response(query)
    answer = response["choices"][0]["message"]["content"]
    tokens_used = response["usage"]["total_tokens"]
    # %%

    pinecone_manager.delete_index()
    # %%

    import random
    VECTOR_DIM = EMBED_DIMENSION
    VECTOR_COUNT = 200
    # Generate data and upsert vectors
    example_data_generator = map(lambda i: (f'id-{i}', [random.random() for _ in range(VECTOR_DIM)]), range(VECTOR_COUNT))
    pinecone_manager.upsert_vectors(example_data_generator)
    # %%


    # Load some text data and upsert vectors
    trec = load_dataset('trec', split='train[1000:1200]')
    text_vector_manager.upsert_text_vectors(trec['text'])
    # %%

