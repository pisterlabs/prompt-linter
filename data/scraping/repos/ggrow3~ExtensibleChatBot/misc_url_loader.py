from langchain.document_loaders import UnstructuredURLLoader
import session_info

from chatbot_settings import ChatBotSettings
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

chatbotSettings = ChatBotSettings()

session_info.show()

urls = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]

print(urls)

loader = UnstructuredURLLoader(urls=urls)

print(loader)

data = loader.load()

print(data)

