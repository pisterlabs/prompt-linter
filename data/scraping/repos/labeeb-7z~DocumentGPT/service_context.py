from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, LLMPredictor
from langchain import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI

import embeddings.prompt_helper as prompt_helper


def get_service_context(embed="HF", llm="OpenAI"):

    print("embed",embed)
    print("llm",llm)

    embed_model = LangchainEmbedding(HuggingFaceHubEmbeddings()) if embed == "HF" else LangchainEmbedding(OpenAIEmbeddings())

    llm_predictor =  LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=512)) if llm == "OpenAI" else LLMPredictor(llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.4,"max_length":256}))
  
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, chunk_size_limit=512)


    return service_context
