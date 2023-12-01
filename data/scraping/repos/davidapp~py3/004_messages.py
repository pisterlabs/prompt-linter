from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

text = "What would be a good company name for a company that makes colorful socks?"
messages = [HumanMessage(content=text)]
# HumanMessage: A ChatMessage coming from a human/user.
# AIMessage: A ChatMessage coming from an AI/assistant.
# SystemMessage: A ChatMessage coming from the system.
# FunctionMessage: A ChatMessage coming from a function call.

chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

print("HumanMessage for chat_models")
print(chat_model.predict_messages(messages))