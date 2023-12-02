from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()


class TokenPrinter(BaseCallbackHandler):
    # This tells the method that we will call it every time the LLM returns us a new token.
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler
chat = ChatOpenAI(
    max_tokens=25,
    streaming=True,
    callbacks=[TokenPrinter()],
)

message = HumanMessage(content="Tell me a joke")

chat([message])
