import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
import tiktoken

# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

Our_Text = "Hello my name is quinton and i am working as a business analyst"
text_embedding = embeddings.embed_query(Our_Text)
print(f"Our embedding is length {text_embedding}")