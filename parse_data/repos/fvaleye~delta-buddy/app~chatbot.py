import logging
from typing import Optional

import torch
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Databricks, HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import pipeline

from app.config import CHROMA_SETTINGS, ExecutionContext, config
from app.consts import PROMPT_FORMAT
from app.databricks_utils.manager import DatabricksManager
from app.models import Answer


class ChatBot:
    def __init__(
        self, execution_context: ExecutionContext = ExecutionContext.LOCAL
    ) -> None:
        self.execution_context = execution_context
        if self.execution_context.value == ExecutionContext.LOCAL.value:
            logging.info(
                "Downloading and loading the QA chain, this may take a long time..."
            )
            embeddings = HuggingFaceEmbeddings(model_name=config.PREPARATION_MODEL_NAME)

            self.db = Chroma(
                persist_directory=config.PERSIST_DIRECTORY,
                embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS,
            )
            self.reset_context()
            logging.info("The QA chain is loaded.")
        elif self.execution_context.value == ExecutionContext.DATABRICKS.value:
            self.databricks_job_manager = DatabricksManager(
                llm=Databricks(
                    host=config.DATABRICKS_SERVER_HOSTNAME,
                    cluster_id=config.DATABRICKS_CLUSTER_ID,
                    cluster_driver_port=config.DATABRICKS_LLM_PORT,
                )
            )
            self.serving_mode = config.DATABRICKS_SERVING_MODE

    def build_qa_chain(self):
        torch.cuda.empty_cache()
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_FORMAT,
        )

        instruct_pipeline = pipeline(
            model=config.DATABRICKS_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
            max_new_tokens=1024,
            top_p=0.95,
            top_k=50,
        )
        hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
        logging.info("loading chain, this can take some time...")
        return load_qa_chain(
            llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True
        )

    def reset_context(self):
        self.qa_chain = self.build_qa_chain()

    def get_similar_docs(self, question: str, similar_doc_count: int):
        return self.db.similarity_search(
            question, include_metadata=True, k=similar_doc_count
        )

    def chat(
        self, question: str, from_databricks_notebook: bool = False
    ) -> Optional[Answer]:
        try:
            if self.execution_context.value == ExecutionContext.DATABRICKS.value:
                if (
                    self.serving_mode.value
                    == config.DATABRICKS_SERVING_MODE.NOTEBOOK_HOSTED_API.value
                ):
                    logging.info(
                        "From Databricks context using the LLM to answer the question."
                    )
                    answer = self.databricks_job_manager.llm(
                        prompt=question,
                        stop=list(),
                    )
                    return Answer(question=question, answer=answer.strip().capitalize())
                elif (
                    self.serving_mode.value
                    == config.DATABRICKS_SERVING_MODE.NOTEBOOK_API.value
                ):
                    logging.info(
                        "From Databricks context using job API to run the notebook to answer the question."
                    )
                    return self.databricks_job_manager.submit_notebook_job_to_cluster(
                        question=question
                    )
            elif self.execution_context.value == ExecutionContext.LOCAL.value:
                logging.info("Loading the QA chain to provide an answer.")
                similar_docs = self.get_similar_docs(
                    question, similar_doc_count=config.SOURCE_DOCUMENTS_MAX_COUNT
                )
                result = self.qa_chain(
                    {"input_documents": similar_docs, "question": question}
                )
                answer = result["output_text"]
                for document in result["input_documents"]:
                    source_id = document.metadata["source"]
                    answer += f"\n (Source: {source_id})"
                return (
                    Answer(question=question, answer=answer.strip().capitalize())
                    if not from_databricks_notebook
                    else Answer.to_html(
                        question=question, answer=answer.strip().capitalize()
                    )
                )

            raise ValueError(
                f"This execution context is not supported {self.execution_context}"
            )
        except Exception as exception:
            logging.exception("An error occurred while answering the question.")
            raise exception
