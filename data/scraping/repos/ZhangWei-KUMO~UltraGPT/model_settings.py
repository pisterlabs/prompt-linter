import openai
from langchain.chat_models import ChatOpenAI

from agent_backend.schemas import ModelSettings
from agent_backend.settings import settings
from agent_backend.web.api.agent.api_utils import rotate_keys
# if(settings.openai_api_base == "<Should be updated via env>"):
#     print("openai_api_base未配置")
# if(settings.db_user == "<Should be updated via env>"):
#     print("数据库USER未配置")
# if(settings.db_pass == "<Should be updated via env>"):
#     print("数据库密码未配置")
# if(settings.db_base == "<Should be updated via env>"):
#     print("数据库名称未配置")
# if(settings.vector_db_url == "<Should be updated via env>"):
#     print("向量数据库URL未配置")
# if(settings.vector_db_api_key == "<Should be updated via env>"):
#     print("向量数据库API KEY未配置")

openai.api_base = settings.openai_api_base

def create_model(model_settings: ModelSettings, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        client=None,  # Meta private value but mypy will complain its missing
        openai_api_key=rotate_keys(
            primary_key=settings.openai_api_key,
            secondary_key=settings.secondary_openai_api_key,
        ),
        openai_api_base=settings.openai_api_base,
        temperature=model_settings.temperature,
        model=model_settings.model,
        max_tokens=model_settings.max_tokens,
        streaming=streaming
    )
