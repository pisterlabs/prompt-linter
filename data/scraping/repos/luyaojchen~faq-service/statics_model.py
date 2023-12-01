from enum import Enum
from pathlib import Path

import flask
import openai
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper, Document, \
    OpenAIEmbedding
from services.environment_service import EnvService

openai.openai_api_key = EnvService.get_openai_api_key()
openai.openai_organization = EnvService.get_openai_organization()
openai.organization = EnvService.get_openai_organization()

prompt_helper = PromptHelper(3900, 256, 20)
# for now keep in memory of all documents in list
g_index = {}


file_extensions_mappings = {
    "application/pdf": ".pdf",
    "application/msword": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "text/html": ".html",
    # Epub
    "application/epub+zip": ".epub",
}

class LLMPredictorFAQ(Enum):
    GPT3 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    GPT4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4"))

class EmbedModelFAQ(Enum):
    ADA = OpenAIEmbedding()

    @staticmethod
    def get_last_token_usage():
        return EmbedModelFAQ.ADA.value.last_token_usage

class Models(Enum):
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"

    # Map these models to the LLMPredictor ones
    @staticmethod
    def get_llm_predictor(model: str):
        if model == Models.GPT3.value:
            return LLMPredictorFAQ.GPT3.value
        elif model == Models.GPT4.value:
            return LLMPredictorFAQ.GPT4.value
        else:
            raise ValueError("Invalid model")

    @staticmethod
    def get_last_token_usage(model: str):
        return Models.get_llm_predictor(model).last_token_usage

    def __str__(self):
        return self.value

    @staticmethod
    def get_models():
        return [model.value for model in Models]

class Answer:
    def __init__(self, answer_id, answer_text):
        self.answer_id = answer_id
        self.answer_text = answer_text


class ResponseStatics:

    @staticmethod
    def build_api_error(error_message):
        response = flask.jsonify({"error": error_message})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @staticmethod
    def build_compose_success_message(tokens):
        response = flask.jsonify({"message": "Successfully composed knowledgebase", "tokens_used": tokens})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @staticmethod
    def build_upload_success_message():
        response = flask.jsonify({"message": "Successfully uploaded document"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @staticmethod
    def build_answer_response(answer: Answer):
        response = flask.jsonify({"answer_id": answer.answer_id, "answer_text": answer.answer_text})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @staticmethod
    def build_creation_response(knowledgebase_id):
        response = flask.jsonify({"knowledgebase_id": knowledgebase_id})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @staticmethod
    def build_list_response(knowledgebase_ids):
        response = flask.jsonify({"knowledgebase_ids": knowledgebase_ids})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

