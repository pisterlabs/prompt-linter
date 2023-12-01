from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from google.auth.credentials import CredentialsWithQuotaProject
import os
load_dotenv()  # take environment variables from .env.

chat = ChatOpenAI(
    model_name=os.getenv("OPENAI_CHAT_MODEL_NAME"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
print(chat(messages))
