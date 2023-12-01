from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever


def initTemporaryChroma() -> Chroma: 
    # 加载文件夹中所有的 txt 文件
    dir_path = "./src/doc/qa/"
    loader = DirectoryLoader(dir_path, glob='**/*.txt')
                             
    # 将数据转成 document 对象
    documents = loader.load()
    
    # 初始化加载器
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)
    
    # 初始化 openAI 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    # 将 document 通过 openAI 的 embeddings 对象计算 embedding 向量信息，并临时存储 Chroma 向量数据库中
    return Chroma.from_documents(split_docs, embeddings)

def question(query): 
    # 初始化向量数据库
    docsearch = initTemporaryChroma()
    
    # 创建问答对象    
    retriever = VectorStoreRetriever(vectorstore=docsearch)          
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever, chain_type="stuff", return_source_documents=True)
    
    # 提问
    return qa({"query": query})

