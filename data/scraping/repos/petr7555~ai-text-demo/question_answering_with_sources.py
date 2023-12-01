from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate

from ai_text_demo.constants import WEAVIATE_URL, OPENAI_API_KEY
from ai_text_demo.utils import relative_path_from_file

DATA_PATH = relative_path_from_file(__file__, "data/state_of_the_union.txt")

with open(DATA_PATH) as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

docsearch = Weaviate.from_texts(
    texts,
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
    weaviate_url=WEAVIATE_URL,
    by_text=False,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
)

print("Question answering with single source")
print(chain(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
))

print("Question answering with multiple sources")
print(chain(
    {"question": "Who is the president?"},
    return_only_outputs=True,
))
