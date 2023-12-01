
import openai, os
import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser

openai.api_key = ""

text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
parser = SimpleNodeParser(text_splitter=text_splitter)
documents = SimpleDirectoryReader('./data/faq/').load_data()
nodes = parser.get_nodes_from_documents(documents)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
))
service_context = ServiceContext.from_defaults(embed_model=embed_model)

dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)
index = GPTFaissIndex(nodes=nodes,faiss_index=faiss_index, service_context=service_context)



from llama_index import QueryMode

openai.api_key = os.environ.get("OPENAI_API_KEY")

query_engine = index.as_query_engine()
response = query_engine.query(
    "鲁迅先生在日本学习医学的老师是谁?",
    mode=QueryMode.EMBEDDING,
    verbose=True,)
print(response)
