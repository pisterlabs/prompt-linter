import os
import pickle
from typing import Tuple

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI

from utils import load_config


class AnsweringModel:
    def __init__(self):
        load_dotenv()
        config_data = load_config("DATASET")
        config_model = load_config("MODEL")
        vectorstore_path = os.path.join(config_data.get("DATA_PATH"), config_data.get("VECTORSTORE_FILENAME"))

        with open(vectorstore_path, "rb") as f:
            self.store = pickle.load(f)

        self.model = OpenAI(temperature=config_model.get("TEMPERATURE"),
                            max_tokens=config_model.get("MAX_TOKENS"),
                            model_name=config_model.get("MODEL_NAME"))

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(chain_type=config_model.get("CHAIN_TYPE"),
                                                                 llm=self.model,
                                                                 retriever=self.store.as_retriever())

    def answer(self, query: str) -> Tuple[str, str]:
        result = self.chain({"question": query})
        return result['answer'].strip(), result['sources']
