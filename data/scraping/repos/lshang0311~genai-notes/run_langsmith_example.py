from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
response = llm.predict("Hello!")
print(response)
