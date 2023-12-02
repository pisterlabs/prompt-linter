from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

from api.settings import chat as chat_settings


def add_langserve_routes(app: FastAPI):
    settings = chat_settings.ChatSettings()
    prompt = ChatPromptTemplate.from_template("{topic} に関するジョークをルー大柴風に答えてください。")
    model = ChatOpenAI(
        model_name=settings.openai_model_name,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        max_retries=1,
        model_kwargs={
            "deployment_id": settings.openai_deployment_id,
            "api_type": settings.openai_api_type,
            "api_version": settings.openai_api_version,
        },
    )

    add_routes(
        app,
        prompt | model,
        path="/langserve/openai",
    )
