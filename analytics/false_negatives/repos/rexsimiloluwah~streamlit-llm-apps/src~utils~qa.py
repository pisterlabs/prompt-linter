from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import PromptTemplate

class SimpleDocumentQA:
    def __init__(self, openai_api_key: str, chain_type: str = "stuff"):
        self._chain_type = chain_type 
        self._llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            temperature=0
        )
        self._db = None 
    
    def load_db(self, docs):
        print("[INFO] Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        print("[INFO] Successfully created embeddings.")

        self._db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        
    def load_qa_chain(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(docs)

        if not self._db:
            self.load_db(docs)
        
        retriever = self._db.as_retriever(search_type="similarity")

        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

        qa_chain = RetrievalQA.from_chain_type(
            self._llm,
            retriever=retriever, 
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        return qa_chain
