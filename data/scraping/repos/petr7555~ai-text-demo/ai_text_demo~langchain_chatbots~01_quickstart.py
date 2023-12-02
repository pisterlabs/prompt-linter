import dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

# Load OPENAI_API_KEY
dotenv.load_dotenv()

print("Plain chat model")
chat = ChatOpenAI()
print(chat([
    HumanMessage(content="Translate this sentence from English to French: I love programming.")
]))

print("Plain chat model - multiple messages")
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
print(chat(messages))

print("ConversationChain")
conversation = ConversationChain(llm=chat)
print(conversation.run("Translate this sentence from English to French: I love programming."))
print(conversation.run("Translate it to German."))
