from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import os
import asyncio
import threading

lock = threading.Lock()
class Ai:
    """
    ai接口
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key="**************************************",
            openai_api_base="*******************************",
            model_name="gpt-3.5-turbo",
            streaming=True,
        )
        self.embeddings = OpenAIEmbeddings(
            deployment_name="text-embedding-ada-002",
            openai_api_key="****************************",
            openai_api_base="***********************************",
            openai_api_type="azure",
            chunk_size=1,
            openai_api_version="2023-06-01-preview",
        )


class Fdb:
    """
    数据库
    """
    def __init__(self, embeddings,path):
        self.embeddings = embeddings
        self.path = path
        try:
            self.db = FAISS.load_local("db", self.embeddings)
        except:
            print("数据库不存在，正在初始化数据库......")
            self.fresh()


    def add(self, load_path):
        loader = PyPDFLoader(load_path)
        texts = loader.load_and_split()
        lock.acquire() # 上锁
        self.db.add_documents(texts)
        self.db.save_local("db")
        lock.release()  # 解锁
        pass

    def fresh(self):
        files = os.listdir(self.path)
        self.db = FAISS.from_texts(["初始化数据库"], embedding=self.embeddings)
        os.system("rm -rf db")
        while os.path.exists("db") and os.path.isdir("db"):
            pass
        if len(files) == 0:
            self.db.save_local("db")
        else:
            for file in files:
                filepath = os.path.join(self.path, str(file))
                self.add(filepath)
        pass

    def search(self,query):
        retriever = self.db.as_retriever()
        re_doc = retriever.get_relevant_documents(query)
        return re_doc


class File:
    """
    文件
    """
    def __init__(self):
        self.ALLOWED_EXTENSIONS = {'pdf'}

    def _allowed_file(self,filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def add_file(self,file,load_path):
        if self._allowed_file(file.filename):
            lock.acquire()  # 上锁
            file.save(load_path)
            lock.release()  # 解锁
        else:
            raise RuntimeError('文件格式错误，只能上传pdf文件')
        pass

    def check_file(self,folder):
        files = os.listdir(folder)
        if len(files) == 0:
            return {'num': 0, 'files': []}
        else:
            return {'num': len(files), 'files': files}
        pass


    def delete_file(self,filename):
        lock.acquire()  # 上锁
        os.remove(filename)
        lock.release()  # 解锁
        pass



class Chat_with_file:
    """
    文件聊天
    """
    def __init__(self,llm,retriever):
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
        )

class Chat_without_file:
    """
    聊天
    """
    def __init__(self,llm):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.qa = ConversationChain(

            llm=llm,
            verbose=True,
        )



class MyCustomHandler(BaseCallbackHandler):
    def __init__(self,queue):
        self.queue = queue
    def on_llm_new_token(self, token: str, **kwargs):
        # sys.stdout.write(token)
        # sys.stdout.flush()
        self.queue.put(token)


