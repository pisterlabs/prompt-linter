import logging
import os
from dotenv import load_dotenv
load_dotenv()

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY=os.environ["SERPAPI_API_KEY"]

from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)


output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)