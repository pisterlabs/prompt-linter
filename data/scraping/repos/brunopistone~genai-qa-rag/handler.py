from __future__ import annotations
import boto3
import json
from langchain import SagemakerEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.summary import SummarizerMixin
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import BaseChatMessageHistory, BaseRetriever, Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch
import logging
import os
from pydantic import Field
import traceback
from typing import Any, Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)
if len(logging.getLogger().handlers) > 0:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

bucket = os.getenv("s3_bucket", default=None)
config_file = os.getenv("genai_configs", default=None)

s3_client = boto3.client('s3')

falcon_template = """
    Use the following pieces of context to answer the question at the end. You must not answer a question not related to the documents.
    If you don't know the answer, just say "Unfortunately, I can't help you with that", don't try to make up an answer.
    
    {chat_history} 
    
    {context}
    
    Question: {question}
    Detailed Answer:
"""

def read_configs(s3_bucket, file_path):
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=file_path)

        config = yaml.safe_load(response["Body"])

        return config
    except yaml.YAMLError as e:
        stacktrace = traceback.format_exc()
        logger.error(stacktrace)

        return e

class Chain:
    def __init__(self, embedding_endpoint, llm_endpoint):
        self.embedding_endpoint = embedding_endpoint
        self.llm_endpoint = llm_endpoint
        self.embeddings = None
        self.vector_search = None
        self.retriever = None
        self.llm = None
        self.memory = None

    def build(self, config, history=[]):
        region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

        embedding_endpoint_name = config["embeddings"][self.embedding_endpoint]["endpoint_name"]
        embedding_handler = eval(config["embeddings"][self.embedding_endpoint]["content_handler"])()

        self.embeddings = SagemakerEndpointEmbeddings(
            endpoint_name=embedding_endpoint_name,
            region_name=region,
            content_handler=embedding_handler
        )

        self.vector_search = OpenSearchVectorSearch(
            opensearch_url=config["es_credentials"]["endpoint"],
            index_name=config["es_credentials"]["index"],
            embedding_function=self.embeddings,
            http_auth=(config["es_credentials"]["username"], config["es_credentials"]["password"])
        )

        self.retriever = DocumentRetrieverExtended(
            self.vector_search,
            "embedding",
            "passage",
            k=config["llms"][self.llm_endpoint]["query_results"],
        )

        llm_endpoint_name = config["llms"][self.llm_endpoint]["endpoint_name"]
        llm_handler = eval(config["llms"][self.llm_endpoint]["content_handler"])()
        llm_model_kwargs = config["llms"][self.llm_endpoint]["model_kwargs"]

        self.llm = SagemakerEndpoint(
            endpoint_name=llm_endpoint_name,
            region_name=region,
            model_kwargs=llm_model_kwargs,
            content_handler=llm_handler
        )

        self.memory = ConversationBufferWindowMemoryExtended(
            k=config["llms"][self.llm_endpoint]["memory_window"],
            chat_memory=history,
            memory_key="chat_history",
            return_messages=True)

class ChatbotChain(Chain):
    def __init__(self, embedding_endpoint, llm_endpoint):
        logger.info("Building ChatbotChain")

        super().__init__(embedding_endpoint, llm_endpoint)
        self.replace_strings = [
            {
                "key": "The AI keeps the answer conversational and provides lots of specific details from its context.",
                "value": "The AI keeps the answer conversational and provides lots of specific details from its context. Please keep the answer in 50 words or less."
            },
            {

                "key": "Use the following pieces of context to answer the question at the end.",
                "value": "Use the following pieces of context to answer the question at the end. Please keep the answer in 50 words or less."
            }
        ]

    def build(self, config, history=[]):
        super().build(config, history)

        prompt_template = eval(config["llms"][self.llm_endpoint]["template"])

        for item in self.replace_strings:
            prompt_template = prompt_template.replace(item["key"], item["value"])

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "chat_history"]
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            memory=self.memory
        )

        return qa

class ChatQAChain(Chain):
    def __init__(self, embedding_endpoint, llm_endpoint):
        logger.info("Building ChatQAChain")

        super().__init__(embedding_endpoint, llm_endpoint)

    def build(self, config, history=[]):
        super().build(config, history)

        prompt_template = eval(config["llms"][self.llm_endpoint]["template"])

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question", "chat_history"]
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True,
            memory=self.memory
        )

        return qa

class BaseChatMemoryExtended(BaseChatMemory):

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                if not "answer" in outputs.keys():
                    raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = "answer"
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

class ConversationBufferWindowMemoryExtended(ConversationBufferWindowMemory, BaseChatMemoryExtended):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConversationSummaryMemoryExtended(ConversationSummaryMemory, BaseChatMemoryExtended, SummarizerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DocumentRetrieverExtended(BaseRetriever):
    def __init__(self, retriever, vector_field, text_field, k=3, return_source_documents=False, score_threshold=None, **kwargs):
        self.k = k
        self.vector_field = vector_field
        self.text_field = text_field
        self.return_source_documents = return_source_documents
        self.retriever = retriever
        self.filter = filter
        self.score_threshold = score_threshold
        self.kwargs = kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = []

        docs = self.retriever.similarity_search_with_score(query, k=self.k, vector_field=self.vector_field, text_field=self.text_field, **self.kwargs)

        if docs:
            for doc in docs:
                metadata = doc[0].metadata
                metadata["score"] = doc[1]
                if self.score_threshold is None or \
                        (self.score_threshold is not None and metadata["score"] >= self.score_threshold):
                    results.append(Document(
                        page_content=doc[0].page_content,
                        metadata=metadata
                    ))

        return results

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await super().aget_relevant_documents(query)

class FalconHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps({"inputs": prompt,
                                "parameters": model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        ans = res[0]['generated_text'][self.len_prompt:]
        ans = ans[:ans.rfind("Human")].strip()
        return ans

class GPTJHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = {'text_inputs': prompt, **model_kwargs}
        return json.dumps(input_str).encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        results = output.read().decode("utf-8")
        response = json.loads(results)
        return response["embedding"]

def lambda_handler(event, context):
    try:

        config = read_configs(bucket, config_file)

        logger.info(event)

        user = event["user"]
        question = event["question"]
        chat_memory = event["chat_memory"]
        llm_endpoint = event["llm_endpoint"]
        embeddings_endpoint = event["embeddings_endpoint"]
        selected_type = event["selected_type"]

        history = ChatMessageHistory()

        for message in chat_memory:
            history.add_user_message(message[0])
            history.add_ai_message(message[1])

        if user != "":
            config["es_credentials"]["index"] = config["es_credentials"]["index"] + "-" + user

        if selected_type == "Chat Q&A":
            chain = ChatQAChain(embeddings_endpoint, llm_endpoint)
        else:
            chain = ChatbotChain(embeddings_endpoint, llm_endpoint)

        qa = chain.build(config, history)

        sources = []

        answer = qa({"question": question, "chat_history": chat_memory})

        if len(answer.get("source_documents", [])) > 0:
            for el in answer.get("source_documents"):
                sources.append({
                    "image": el.metadata["image"] if "image" in el.metadata else "",
                    "details": f'Document = {el.metadata["file_name"]} | Page = {el.metadata["page"]} | Score = {el.metadata["score"]}',
                    "passage": (el.page_content[:300] + '..') if len(el.page_content) > 300 else el.page_content
                })

                if len(sources) == 3:
                    break

        if "answer" not in answer and "text" in answer:
            answer["answer"] = answer["text"]

        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "answer": answer.get("answer").strip(),
                    "sources": sources
                }
            )
        }

    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e
