import logging
from domain.service.search_cognitive import SearchCognitive
from domain.service.ask_generate import AskGenerate
from domain.service.clean_text import clean_text
from domain.excepciones.error_del_negocio import ErrorDelNegocio
from infraestructure.adapter.cognitive_search_adapter import (
    CognitiveSearchAdapter,
)
from infraestructure.adapter.openai_adapter import OpenAIAdapter

from domain.modelo.output_data_dto import OutputDataDto

query_cs = SearchCognitive()
adapter_cs = CognitiveSearchAdapter()
query_gpt = AskGenerate()
adapter_gpt = OpenAIAdapter()
error_del_negocio = ErrorDelNegocio()


class HandlerQuery:
    def execute(self, input_data: dict) -> OutputDataDto:
        """
        execute HandlerQuery
        Args: input_data
        Returns: query.execute_service
        """
        logging.warning(__name__)
        logging.warning(f"input_data: {input_data}")
        question = input_data["question"]
        cognitive_response = query_cs.execute_service(question, adapter_cs)

        text_to_gpt = cognitive_response["text"]
        indexed_document = cognitive_response["document"]

        gpt_response = query_gpt.execute_service(
            question, text_to_gpt, "question", adapter_gpt
        )
        indexed_document = clean_text(indexed_document)
        indexed_document = indexed_document.split(",")

        logging.warning("====================")
        logging.warning("buscamos el texto indexado: ")
        logging.warning(cognitive_response)
        logging.warning("====================")

        logging.warning("RESPUESTA DE GPT-3")
        logging.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        logging.warning(f"de acuerdo a {indexed_document}")
        logging.warning(gpt_response)
        logging.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        return OutputDataDto(response=gpt_response, document=indexed_document)
