####### 2. CREATE & PERSIST EMBEDDINGS FROM DOCUMENT CHUNKS
###Take a document chunk and create vectors embeddings. Similar pieces of text would have similar vectors embeddings. This aids in information retrieval.

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def store_embeddings(document_chunks, remove_prev_db=False):
  embeddings = OpenAIEmbeddings()
  persist_directory = 'docs/chroma/'
  if remove_prev_db:
    #remove old database files if any
    shutil.rmtree(persist_directory, ignore_errors=True)
  vectordb = Chroma.from_documents(documents=document_chunks,
                                   embedding=embeddings,
                                   persist_directory=persist_directory)
  return vectordb


### Test vectordb
def test_vectordb(vectordb, test_prompts):
  print("VectorDB Collection Count : " + str(vectordb._collection.count()))
  for test_prompt in test_prompts:
    print("Prompt given : " + test_prompt)
    print("Finding documents using vector similarity search")
    # Semantic Similarity Search
    docs = vectordb.similarity_search(test_prompt, k=3)
    print(len(docs))
    for doc in docs:
      print(doc.metadata)
      print(doc.page_content)
