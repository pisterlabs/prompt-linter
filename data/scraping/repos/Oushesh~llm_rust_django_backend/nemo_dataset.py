"""
nemo_dataset with Guardrails:
reduced toxicity.
"""

from typing import Any, Callable, Coroutine
from langchain.llms.base import BaseLLM
from nemoguardrails import LLMRails, RailsConfig
from django.conf import settings

import yaml,os
from dotenv import load_dotenv

load_dotenv(os.path.join(settings.BASE_DIR,"hallucination_app","Guardrails",".env"))

#load_dotenv(".env")

from hallucination_app.Guardrails.knowledge_base.constants import model_content, rag_colang_content
from hallucination_app.Guardrails.utils import test_colang_config,test_model_config

def guardrail(prompt:str)->bool:
    assert (isinstance(prompt, str) == True)
    try:
        import llama_index
        from llama_index.indices.query.base import BaseQueryEngine
        from llama_index.response.schema import StreamingResponse

    except ImportError:
        raise ImportError(
            "Could not import llama_index, please install it with "
            "`pip install llama_index`."
        )

    #TODO: add testing function to test the contents
    #test_colang_config(rag_colang_content)
    test_model_config(model_content)

    # initialize rails config
    config = RailsConfig.from_content(
        colang_content=rag_colang_content,
        yaml_content=model_content
    )

    app = LLMRails(config)

    def _get_llama_index_query_engine(llm: BaseLLM):
        docs = llama_index.SimpleDirectoryReader(
            input_files=[os.path.join(settings.BASE_DIR,"hallucination_app","Guardrails","knowledge_base","report.md")]
        ).load_data()
        llm_predictor = llama_index.LLMPredictor(llm=llm)
        index = llama_index.GPTVectorStoreIndex.from_documents(
            docs, llm_predictor=llm_predictor
        )
        default_query_engine = index.as_query_engine()
        return default_query_engine

    def _get_callable_query_engine(
        query_engine: BaseQueryEngine,
    ) -> Callable[[str], Coroutine[Any, Any, str]]:
        async def get_query_response(query: str) -> str:
            response = query_engine.query(query)
            if isinstance(response, StreamingResponse):
                typed_response = response.get_response()
            else:
                typed_response = response
            response_str = typed_response.response
            if response_str is None:
                return ""
            return response_str

        return get_query_response

    query_engine = _get_llama_index_query_engine(app.llm)
    app.register_action(
        _get_callable_query_engine(query_engine), name="llama_index_query"
    )

    history = [{"role": "user", "content": prompt}]
    result = app.generate(messages=history)
    print ("result",result)
    return result["content"]


if __name__ == "__main__":
    #test_yaml_file("knowledge_base/model_config.yaml")
    result=guardrail()
    print ("result", result)
