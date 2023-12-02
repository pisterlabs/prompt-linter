import json
import logging
import os
from typing import Any, Dict, List

from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DeepLake

logger = logging.getLogger("uranium")
logger.setLevel(logging.INFO)


class Handler(BaseHandler):
    def __init__(self, model_dir: str = "./deploy/model") -> None:
        super().__init__()
        self.config = self.load_hyperparams(model_dir)
        self.embedding_model = OpenAIEmbeddings()
        self.vectorstore = self.load_vectorstore(
            model_dir,
            self.embedding_model,
        )
        self.model_runner = OpenAiModelRunner(
            api_key=os.environ.get("OPENAI_API_KEY", None),
            project_name=self.project_name,
        )
        self.prompt = PromptTemplate(
            template=fetch_qa_template(), input_variables=["question", "summaries"]
        )
        self.chain = load_qa_with_sources_chain(
            OpenAI(
                temperature=self.config["temperature"],
                model_name=self.config["llm_model_name"],
            ),
            chain_type=self.config["chain_type"],
            prompt=self.prompt,
        )

    def serve_inference(self, event: dict, context: dict) -> dict:
        event_body = event.get("body", None)
        if event_body:
            event = json.loads(event_body)
        prompt = event.get("prompt", None)
        self.docs = self.vectorstore.similarity_search(prompt, k=3)
        sources = [doc.metadata["source"] for doc in self.docs]
        sources = [source.split("/")[-1].split(".")[0] for source in sources]
        result = self.chain(
            {"input_documents": self.docs, "question": prompt},
        )
        return {"result": result["output_text"]}

    @staticmethod
    def load_hyperparams(model_dir: str) -> dict:
        with open(os.path.join(model_dir, "config.json")) as f:
            hparams = json.load(f)
        logger.info(json.dumps(hparams, indent=4))
        return hparams

    @staticmethod
    def load_vectorstore(model_dir: str, embedding_model):
        dataset_dir = os.path.join(model_dir, "articles")
        vectorstore = DeepLake(
            dataset_path=dataset_dir,
            embedding_function=embedding_model,
            read_only=True,
        )
        return vectorstore


handler = Handler()


def handle_request(event: dict, context: dict) -> dict:
    return handler.handle_request(event, context)
