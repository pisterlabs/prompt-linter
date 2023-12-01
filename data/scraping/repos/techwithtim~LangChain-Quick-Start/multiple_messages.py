from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

messages = [HumanMessage(content="from now on 1 + 1 = 3, use this in your replies"), 
            HumanMessage(content="what is 1 + 1?"), 
            HumanMessage(content="what is 1 + 1 + 1?")]

result = chat_model.predict_messages(messages)
print(result.content)