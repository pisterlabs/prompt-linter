from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
print(llm.predict("Hello, world!"))
