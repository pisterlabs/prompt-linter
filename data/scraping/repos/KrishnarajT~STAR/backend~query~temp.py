from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
embedding2  =  HuggingFaceEmbeddings()
vdb_chunks_HF  =   FAISS.load_local("vdb_chunks_HF", embedding2, index_name="indexHF")
ans=vdb_chunks_HF.as_retriever().get_relevant_documents("Hatsumi pool table")
print(ans[0].page_content)