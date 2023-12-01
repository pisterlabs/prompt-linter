from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import sentence_transformers
from langchain.document_loaders import TextLoader
#loader = TextLoader("/devdata/home/user/panyongcan/Project/langchain/langchain/hexiang.txt")
loader = TextLoader("/devdata/home/user/panyongcan/Project/langchain/langchain/7.6专精特新企业简介添加企业名称文本.txt")
documents = loader.load()
import ipdb
ipdb.set_trace()
text_splitter = CharacterTextSplitter(separator='\n',chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
#embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name='')
embeddings.client = sentence_transformers.SentenceTransformer("/devdata/home/user/panyongcan/Project/embedding/m3e-base", device="cuda")
db = FAISS.from_documents(docs, embeddings)

query = "合享汇智信息科技集团集团成立于哪一年？"
docs = db.similarity_search(query)
print(docs[0].page_content)
docs_and_scores = db.similarity_search_with_score(query)
docs_and_scores[0]
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
db.save_local("/devdata/home/user/panyongcan/Project/chat_ui/wenda/memory/default")
new_db = FAISS.load_local("/devdata/home/user/panyongcan/Project/chat_ui/wenda/memory/default", embeddings)
docs = new_db.similarity_search(query)
docs[0]
