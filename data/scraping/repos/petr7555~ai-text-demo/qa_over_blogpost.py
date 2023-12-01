import dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

# Load OPENAI_API_KEY
dotenv.load_dotenv()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query("What is Task Decomposition?"))
