from typing import List

from llama_index import GPTVectorStoreIndex, \
    LLMPredictor, PromptHelper, Document
from langchain.chat_models import ChatOpenAI
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()


class Indexer:
    def __init__(self):
        self.s3 = s3fs.S3FileSystem(key=os.getenv('S3_ACCESS_KEY'), secret=os.getenv('S3_SECRET_KEY'))

    def index_documents(self, documents: List[Document], index_id: str, max_input_size: int = 4096,
                        num_outputs: int = 512,
                        max_chunk_overlap: int = 20, chunk_size_limit: int = 600, temperature: float = 0.7,
                        model_name: str = "gpt-3.5-turbo"):
        prompt_helper = PromptHelper(max_input_size=max_input_size,
                                     num_output=num_outputs,
                                     max_chunk_overlap=max_chunk_overlap,
                                     chunk_size_limit=chunk_size_limit)
        print('Indexing db... ')

        llm_predictor = LLMPredictor(
            llm=ChatOpenAI(temperature=temperature,
                           model_name=model_name,
                           max_tokens=num_outputs)
        )
        print("DB Indexed!")

        index = GPTVectorStoreIndex.from_documents(
            documents,
            llm_predictor=llm_predictor,
            prompt_helper=prompt_helper)

        print("Uploading to S3...")

        index.set_index_id(index_id)
        index.storage_context.persist(os.getenv('BUCKET') + f'/{index_id}', fs=self.s3)
        print("Uploaded to S3!")
        return index_id
