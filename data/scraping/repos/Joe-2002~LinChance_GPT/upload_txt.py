# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import TextLoader

# from sentence_transformers import SentenceTransformer

# # Load the SentenceTransformer model
# model_name = "bert-base-chinese"
# model = SentenceTransformer(model_name)

# class LinFaiss:
#     def __init__(self):
#         self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
#     def load_data(self):
#         files = ['GPT_Project\apps\Lin_FAISS\text\研发简要流程.txt']  # Add more file names if needed

#         documents = []

#         for file in files:
#             my_file = f"./text/{file}"
#             loader = TextLoader(my_file, encoding='utf8')
#             documents.extend(loader.load())

#         return documents
#     # def load_data(self):
#     #     files = ['研发简要流程.txt']  # 如果需要，可以添加更多文件名
#     #     documents = []

#     #     for file in files:
#     #         my_file = f"./text/{file}"
#     #         try:
#     #             with open(my_file, 'r', encoding='utf8') as f:
#     #                 document_content = f.read()
#     #                 documents.append(Document(page_content=document_content))
#     #         except Exception as e:
#     #             print(f"Error loading {my_file}: {e}")

#     #     return documents

#     def get_embeddings(self, docs):
#         vector_store = FAISS.from_documents(docs, embedding=self.embeddings)
#         return vector_store

#     def documents2dict(self, documents):
#         # Convert Document objects to a dictionary format
#         documents_dict = [
#             {'page_content': document.page_content, 'metadata': document.metadata}
#             for document in documents
#         ]
#         return documents_dict

#     def save_vec_data(self, index="apps/Lin_FAISS/faiss_index"):
#         docs = self.load_data()
#         vector_store = self.get_embeddings(docs)
#         vector_store.save_local(index)

#     def get_similarity_documents(self, q, index="apps/Lin_FAISS/faiss_index", limit=3):
#         db = FAISS.load_local(index, self.embeddings)
#         query_embedding = self.embeddings.embed_text(q)
#         docs = db.similarity_search(query_embedding, k=limit)
#         texts = self.documents2dict(docs)
#         return texts


# if __name__ == "__main__":
#     linfaiss = LinFaiss()
    

#     linfaiss.save_vec_data()
#     query = "如何研发一个产品"
#     similar_doc = linfaiss.get_similarity_documents(query)
#     print(similar_doc)

