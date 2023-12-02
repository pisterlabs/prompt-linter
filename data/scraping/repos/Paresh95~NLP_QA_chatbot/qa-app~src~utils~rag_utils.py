from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.vectorstore import VectorStoreRetriever
from src.utils.model_utils import load_text2text_generation_pipeline
from src.utils.vector_store_utils import FaissConnector
from src.utils.general_utils import read_yaml_config


class RagSystem:
    def __init__(self, config: dict, user_id: int):
        self.user_id = user_id
        self.prompt_template = config["prompt_template"]
        self.prompt_input_variables = config["prompt_input_variables"]
        self.local_model_path = config["local_model_path"]
        self.local_tokenizer_path = config["local_tokenizer_path"]
        self.device = config["device"]
        self.max_new_tokens = config["max_new_tokens"]
        self.hugging_face_embedding_model_path = config[
            "hugging_face_embedding_model_path"
        ]
        self.vector_store_path = config["vector_store_path"]
        self.documents_to_retrieve = config["documents_to_retrieve"]
        self.rerank_documents = config["rerank_documents"]

    def _load_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.prompt_template, input_variables=self.prompt_input_variables
        )
        return prompt

    def _load_llm_pipeline(self) -> HuggingFacePipeline:
        pipe = load_text2text_generation_pipeline(
            self.local_model_path,
            self.local_tokenizer_path,
            self.device,
            self.max_new_tokens,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def _load_retriever(self) -> VectorStoreRetriever:
        db = FaissConnector(
            self.hugging_face_embedding_model_path, self.vector_store_path
        ).load_db()
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "filter": {"user_id": self.user_id},
                "k": self.documents_to_retrieve,
            },
        )
        if self.rerank_documents:
            reordering = LongContextReorder()
            pipeline_compressor = DocumentCompressorPipeline(transformers=[reordering])
            retriever = ContextualCompressionRetriever(
                base_compressor=pipeline_compressor, base_retriever=retriever
            )
        return retriever

    def _build_qa_chain(self) -> ConversationalRetrievalChain:
        prompt = self._load_prompt()
        llm = self._load_llm_pipeline()
        retriever = self._load_retriever()
        memory = ConversationBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        return qa_chain

    def run_query(self, query: str) -> dict:
        qa_chain = self._build_qa_chain()
        results = qa_chain({"question": query})
        return results


if __name__ == "__main__":
    config = read_yaml_config("parameters.yaml")
    user_id = 1
    query = "What age might they leave something to their nephews and nieces?"
    results = RagSystem(config=config, user_id=1).run_query(query)
    print(results["question"])
    print(results["answer"])
