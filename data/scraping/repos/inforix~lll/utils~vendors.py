from langchain.vectorstores import Chroma

class ChromaVectorStore:
  def __init__(self, documents, embedding) -> None:
    self.documents = documents
    self.embedding = embedding
    self.docsearch = Chroma.from_documents(documents=documents, embedding=embedding)

  def get_similiar_docs(self, query, k=3, score=False):
    if score:
      similar_docs = self.docsearch.similarity_search_with_score(query, k=k)
    else:
      similar_docs = self.docsearch.similarity_search(query, k=k)
    
    #print(similar_docs)
    return similar_docs