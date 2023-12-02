from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

chat = ChatOpenAI()

human_msg = [HumanMessage(content="Hey There!")]
content = chat.predict_messages(human_msg)

print(content)
