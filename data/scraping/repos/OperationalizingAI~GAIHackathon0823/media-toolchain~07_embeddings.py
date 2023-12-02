from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

import dotenv

dotenv.load_dotenv()

# embeddings_model = OpenAIEmbeddings()

embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)
print(len(embeddings), len(embeddings[0]))
print(embeddings[0])
