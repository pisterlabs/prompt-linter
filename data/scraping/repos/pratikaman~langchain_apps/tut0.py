from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()
chat_model = ChatOpenAI()

# print(chat_model.predict('Hi how are you?'))
print(llm.predict('Hi how are you?'))
