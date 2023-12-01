import os
import requests

from council.agents import Agent

import dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader

from llama_index_skill import LlamaIndexSkill

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", None) is not None)

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

response = query_engine.query("Where do Gatsby and Daisy meet?")
print(response)

index_skill = LlamaIndexSkill(query_engine)
agent = Agent.from_skill(index_skill, "document index")

# message = "Whose eyes are on the billboard?"
# message = "What are the personalities of Tom and Daisy?"
# message = "What era does the book take place in?"
message = "Who falls in love with Daisy?"

result = agent.execute_from_user_message(message)
print(result.best_message.message)
