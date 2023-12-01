from langchain.chat_models import ChatOpenAI
import dotenv
from pretty_print_callback_handler import PrettyPrintCallbackHandler

dotenv.load_dotenv()

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()

from langchain.schema import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(content="I love programming."),
]

pretty_callback = PrettyPrintCallbackHandler()
chat.callbacks = [pretty_callback]


result = chat(messages)

prompt2 = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(content="I love programming."),
    HumanMessage(content="Please now translate the previous message into German"),
]
result2 = chat(prompt2)
# print(result)
