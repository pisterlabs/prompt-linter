from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings()

embedded_query = embeddings_model.embed_query(
    "What was the name mentioned in the conversation?"
)

# print(embedded_query)
# print(len(embedded_query))

documents = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!",
]

embeddings = embeddings_model.embed_documents(documents)
print(len(embeddings))
print(len(embeddings[0]))
