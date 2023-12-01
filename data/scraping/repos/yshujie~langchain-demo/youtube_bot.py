import os

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 初始化向量数据库
def initTemporaryChroma(video_url) -> Chroma:
    # 加载 youtube 频道
    loader = YoutubeLoader.from_youtube_url(video_url)
    
    # 将数据转成 document 对象
    documents = loader.load()
    
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    
    # 分割 youtube documents
    documents = text_splitter.split_documents(documents)
    
    # 初始化 openAI 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    
    # 通过 openAI 的 embeddings 对象计算 embedding 向量信息，并临时存储 Chroma 向量数据库中
    return Chroma.from_documents(documents, embeddings)
    
def chat():
    # 初始化向量数据库
    vector_store = initTemporaryChroma("https://www.youtube.com/watch?v=Dj60HHy-Kqk")
    # 初始化检索器
    retriever = vector_store.as_retriever()
    
    system_template = """
    Use the following context to answer the user's question.
    If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
    ----------
    {question}
    ----------
    {chat_history}
    """
    
    # 构建初始 message 列表，这里可以理解为 openAI 传入的 messages 参数
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}')   
    ]
    
    # 初始化 prompt 对象
    prompt = ChatPromptTemplate.from_messages(messages)
    
    # 初始化问答链
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.1, max_tokens=2048),
        retriever,
        condense_question_prompt=prompt
    )
    
    chat_history = []
    
    while True:
        question = input('问题：')
        
        # 如果 question 空，则跳过
        if not question:
            continue
        
        # 开始发送问题 chat_history 为必须参数,用于存储对话历史
        result = qa({'question': question, 'chat_history': chat_history})
        chat_history.append((question, result['answer']))
        
        print(result['answer'])     