import os
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT


class Brain:
    def __init__(self, path, llm=OpenAI(temperature=0, model_name="text-davinci-003")):
        self.path = path
        self.text_splitter = CharacterTextSplitter(
            separator=" ", chunk_size=1000, chunk_overlap=0
        )
        self.embeddings = OpenAIEmbeddings()
        self.db = None
        self.loaders = []
        self.chat_history = []
        self.llm = llm
        self.db = Chroma(
            persist_directory=self.path, embedding_function=self.embeddings
        )

    def clear(self):
        self.db.delete_collection()
        self.db = Chroma(
            persist_directory=self.path, embedding_function=self.embeddings
        )

    def save(self):
        self.db.persist()

    def loadFileTXT(self, file):
        self.loaders.append(TextLoader(file))

    def loadFilePDF(self, file):
        self.loaders.append(PyPDFLoader(file))

    def loadFolder(self, folder):
        files = os.listdir(folder)
        for file in files:
            if os.path.isfile(os.path.join(folder, file)) and file.endswith(".pdf"):
                print("Loading PDF: " + file)
                self.loadFilePDF(os.path.join(folder, file))
            elif os.path.isfile(os.path.join(folder, file)) and file.endswith(".txt"):
                print("Loading TXT: " + file)
                self.loadFileTXT(os.path.join(folder, file))

    def index(self):
        documents = []
        for loader in self.loaders:
            documents.extend(loader.load())

        self.texts = self.text_splitter.split_documents(documents)

        texts_final = [doc.page_content for doc in self.texts]
        metadatas = [doc.metadata for doc in self.texts]

        self.db.add_texts(texts=texts_final, metadatas=metadatas)

        self.loaders = []

    def wakeup(self):
        retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        self.conversationalChain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, retriever=retriever
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
        )

        prompt_template = """You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided. You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks. 
        If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer. 
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

            Question: {question}
            =========
            Context: {context}
            =========
            Answer:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def ask(self, query):
        result = self.conversationalChain(
            {"question": query, "chat_history": self.chat_history}
        )
        self.chat_history.append((query, result["answer"]))
        return result["answer"]

    def queryRQA(self, query):
        return self.qa.run(query)

    def queryChain(self, query, results=1):
        docs = self.db.similarity_search(query, k=results)
        inputs = [{"context": doc.page_content, "question": query} for doc in docs]
        return self.chain.apply(inputs)

    def queryQA(self, query):
        qa = load_qa_chain(llm=self.llm, chain_type="stuff")
        docs = self.db.similarity_search(query, top_k=5)
        return qa.run(input_documents=docs, question=query)
