import logging
import os
import requests

from council.agents import Agent
from council.chains import Chain
from council.controllers import BasicController
from council.evaluators import BasicEvaluator
from council.filters import BasicFilter
from council.llm import OpenAILLMConfiguration, OpenAILLM
from council.prompt import PromptBuilder
from council.skills import PromptToMessages, LLMSkill
from council.utils import Option
import dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader

from llama_index_skill import LlamaIndexSkill

import logging

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(threadName)s %(name)s:%(funcName)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
)
logging.getLogger("council").setLevel(logging.WARN)

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", None) is not None)
config = OpenAILLMConfiguration(
    api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo"
)
llm = OpenAILLM(config)


url = (
    "https://github.com/jerryjliu/llama_index/blob/main/examples/gatsby/gatsby_full.txt"
)
filename = url.split("/")[-1]

os.makedirs("gatsby_download", exist_ok=True)

response = requests.get(url)
with open(os.path.join("gatsby_download", filename), "wb") as f:
    f.write(response.content)

documents = SimpleDirectoryReader("gatsby_download").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

context_prompt = PromptToMessages(
    PromptBuilder(
        "Please identify query terms to respond to the following user request {{chat_history.last_message}}"
    )
)
context_query_skill = LLMSkill(
    llm,
    "You are an expert in the Great Gatsby. Identify relevant query terms to search for context in the book.",
    context_messages=context_prompt.to_user_message,
)

index_skill = LlamaIndexSkill(query_engine)
index_prompt = PromptToMessages(
    PromptBuilder(
        "Here are relevant quotes from the book: {{chain_history.last_message}} \nUse this to respond to the following user request {{chat_history.last_message}}"
    )
)
response_skill = LLMSkill(
    llm,
    "You are an expert in the Great Gatsby. Provide a helpful response to the user's question",
    context_messages=index_prompt.to_user_message,
)

chain = Chain(
    "docindex", "document index", [context_query_skill, index_skill, response_skill]
)
agent = Agent(BasicController([chain]), BasicEvaluator(), BasicFilter())

# message="Whose eyes are on the billboard?")
# message="What are the personalities of Tom and Daisy?"
# message="What era does the book take place in?"
# message="What is the significance of the eyes on the billboard?"
message = "What are the key plot events in the book?"

result = agent.execute_from_user_message(message=message)
print(result.best_message.message)
