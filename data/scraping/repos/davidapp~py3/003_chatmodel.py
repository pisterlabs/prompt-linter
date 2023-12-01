from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0613")
print(chat_model.predict("hi!"))
