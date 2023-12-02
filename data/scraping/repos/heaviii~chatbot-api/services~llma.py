import os
import threading
from langchain.vectorstores import VectorStore, Chroma
from langchain.vectorstores.redis import Redis
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain

import pinecone 
import getpass


from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from loguru import logger
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA

from config import Config
from services.AgentFunctionChat import AgentFunctionChat
from services.ThreadedGenerator import ChainStreamHandler, ThreadedGenerator
from tools.knowboxTool import KnowboxTool

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

class LLma:

    def __init__(self) -> None:
        logger.info("init")

   
    def askQuestionPiecone(generator, collection_id, prompt, index_path=Config.INDEX_JSON_PATH):
        pinecone.init(
            api_key='c23851db-0963-4e1b-b550-e61ca6f2b832',  # find at app.pinecone.io
            environment='us-west4-gcp-free'  # next to api key in console
        )
        embeddings = OpenAIEmbeddings()

        index_name = "langchain-demo"

        # if you already have an index, you can load it like this
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

        #GPTPineconeIndex.load_from_disk(index_path,llm=OpenAI())
        
        query = "我家的花园叫什么"
        docs = docsearch.similarity_search(query, include_metadata=True)
        print(docs)
        llm = OpenAI(temperature=0)
        #llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator)]), verbose=True)
        chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
        result = chain.run(input_documents=docs, question=query)
        print(result)
        return result
    
    def askQuestionChroma(generator, collection_id, prompt, index_path=Config.INDEX_JSON_PATH):
        # 初始化 openai 的 embeddings 对象
        #loader = TextLoader(Config.DATA_PATH+"/text.txt")
        #documents = loader.load()
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        #texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        #vectordb = Chroma.from_documents(texts, embeddings)
        #vectordb = Chroma(persist_directory=Config.VECTOR_STORE_PATH, embedding_function=embeddings)
        vectordb = KnowboxTool().init_tool_db()

        #从向量库查询prompt相关文本
        #1.
        # docs = vectordb.similarity_search(prompt,k=2)
        #2.
        #retriever = vectordb.as_retriever(search_type="cosine",search_kwargs={"k":2})# search_type="mmr", k=10, alpha=0.5, beta=0.5)
        #docs = retriever.get_relevant_documents(prompt)
        

        genie = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())
        return genie.run(query=prompt)
    
    def askQuestionChromaStream(generator, collection_id, prompt, index_path=Config.INDEX_JSON_PATH):
        try:
            print("prompt:====",prompt)
            #embeddings = OpenAIEmbeddings()

            #docsearch = Chroma(persist_directory=Config.VECTOR_STORE_PATH, embedding_function=embeddings)
            docsearch = KnowboxTool().init_tool_db()
            llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator)]), verbose=True)
            #llm = OpenAI()
            # 创建问答对象
            qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
            # 进行问答
            result = qa({"query": prompt})
            
            print("result:---",result)
            res_dict = {
            }

            res_dict["source_documents"] = []

            for source in result["source_documents"]:
                res_dict["source_documents"].append({
                    "page_content": source.page_content,
                    "metadata":  source.metadata
                })

            return res_dict

        finally:
            generator.close()
    
    
    def query_index_stream(prompt, collection_id):
        generator = ThreadedGenerator()
        threading.Thread(target=LLma.askQuestionChromaStream, args=(generator, collection_id, prompt)).start()
        return generator
    
    
    def create_index_chromadb(self):
        if os.path.exists(Config.VECTOR_STORE_PATH):
            return True
        print('create_index_chromadb')
         # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs()
        print(split_docs)

        # 初始化 openai 的 embeddings 对象
        embeddings = OpenAIEmbeddings()
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        docsearch = Chroma.from_texts([t.page_content for t in split_docs], embeddings, persist_directory=Config.VECTOR_STORE_PATH)
        res = docsearch.similarity_search(query="我家的花园叫什么",k=2)
        print(res)
        docsearch.persist()
        return True
    
    def create_index_pinecone(self):
         # 加载文件夹中的所有txt类型的文件
        split_docs = self.get_docs()

        # 初始化 openai 的 embeddings 对象
        embeddings = OpenAIEmbeddings()
        # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
        pinecone.init(
            api_key='c23851db-0963-4e1b-b550-e61ca6f2b832',  # find at app.pinecone.io
            environment='us-west4-gcp-free'  # next to api key in console
        )

        index_name = "langchain-demo"
        #pinecone.create_index(index_name, dimension=1536)

        
        #docsearch = Pinecone.from_documents([t.page_content for t in split_docs], embeddings, index_name=index_name)
        docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)
        return True
    
    def get_docs(self):
        loader = DirectoryLoader(Config.DATA_PATH, glob='**/*')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()

        # 初始化加载器
        #text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # 切割加载的 document
        return text_splitter.split_documents(documents)

