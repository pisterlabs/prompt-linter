from src.llms.openai import OpenAIManager
from settings import get_settings

app_settings = get_settings()
from langchain.embeddings.openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(openai_api_key=app_settings.OPENAI_API_KEY)  # type: ignore

OpenAIManager.index_file_from_path(
    file_path=r"data\text.jpg",
    embedding_function=embeddings
)
