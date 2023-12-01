from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

chat = ChatOpenAI(model_name="gpt-3.5-turbo")
message = [
    HumanMessage(
        content="Translate this sentence from English to Polish. I love programming."
    ),
]
ai_response = chat(message)
print(ai_response)
