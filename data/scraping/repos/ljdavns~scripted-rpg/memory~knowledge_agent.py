import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import set_keys
import config

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import LLMChain, OpenAI, VectorDBQA, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, Tool
from langchain.agents.agent import AgentExecutor
from typing import Any, Dict, Optional, Sequence
from langchain.agents.agent_toolkits.vectorstore.prompt import PREFIX, ROUTER_PREFIX
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreToolkit,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools.base import BaseTool
from langchain.llms.base import BaseLLM
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.document_loaders import UnstructuredFileLoader
from datetime import datetime

COMMON_DB_NAME = "common"
def create_custom_agent(
    llm: BaseLLM,
    tools: Sequence[BaseTool],
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    verbose: bool = True,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a vectorstore agent from an LLM and tools."""
    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )

# todo: add chat history to agent
class KnowledgeAgent:
    vector_memory_path: str = config.VECSTORE_PATH_CHROMA
    vector_memory: dict[str, Chroma] = {}

    def __init__(self, vector_memory_path: str = None):
        if vector_memory_path:
            self.vector_memory_path = vector_memory_path
        # try:
        #     self.vector_memory[COMMON_DB_NAME] = self.load_vector_memory()
        # except:
        #     self.vector_memory[COMMON_DB_NAME] = self.create_vector_memory_from_dir(config.DOC_PATH)
        # iterate through all subdirs of vector_memory_path
        for subdir in Path(self.vector_memory_path).iterdir():
            if subdir.is_dir() and subdir.name != COMMON_DB_NAME:
                self.vector_memory[subdir.name] = self.load_vector_memory(subdir.name)

    # def load_vector_memory(self, db_name: str = COMMON_DB_NAME):
    #     print("loading vector memory of {}...".format(db_name))
    #     embeddings = OpenAIEmbeddings()
    #     vector_memory = FAISS.load_local("{}/{}".format(self.vector_memory_path, db_name), embeddings)
    #     return vector_memory
    
    def load_vector_memory(self, db_name: str = COMMON_DB_NAME):
        print("loading vector memory of {}...".format(db_name))
        embeddings = OpenAIEmbeddings()
        vector_memory = Chroma(collection_name=db_name, embedding_function=embeddings, persist_directory="{}/{}".format(self.vector_memory_path, db_name))
        return vector_memory

    # def save_vector_memory(self, doc_id, doc_name):
    #     db_name = "{}-{}".format(doc_id, doc_name) if doc_id else COMMON_DB_NAME
    #     self.vector_memory[db_name].save_local("{}/{}".format(self.vector_memory_path, db_name))

    def save_vector_memory(self, doc_id, doc_name):
        db_name = "{}-{}".format(doc_id, doc_name) if doc_id else COMMON_DB_NAME
        self.vector_memory[db_name].persist()

    def add_text_to_vector_memory(self, doc_id, doc_name, texts: list):
        current_dt = datetime.now().strftime("%Y-%m-%d")
        metadatas = [{"source": 'storybook content of {}({}) in {}'.format(doc_name, doc_id, current_dt)} for i in range(len(texts))]
        db_name = "{}-{}".format(doc_id, doc_name) if doc_id else COMMON_DB_NAME
        if db_name not in self.vector_memory:
            self.vector_memory[db_name] = self.create_vector_memory_from_texts(texts, metadatas)
        else:
            self.vector_memory[db_name].add_texts(texts=texts, metadatas=metadatas)

    def load_file_to_vector_memory(self, doc_id, doc_name, file_path: str, file_name: str):
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "，", "。", "\n"])
        splitted_documents = text_splitter.split_documents(documents)
        for doc in splitted_documents:
            doc.metadata["source"] = file_name
            # doc.metadata["doc_id"] = doc_id
            # doc.metadata["doc_name"] = doc_name
        db_name = "{}-{}".format(doc_id, doc_name) if doc_id else COMMON_DB_NAME
        if db_name not in self.vector_memory:
            # self.vector_memory[db_name] = FAISS.from_documents(splitted_documents, OpenAIEmbeddings())
            self.vector_memory[db_name] = Chroma.from_documents(splitted_documents, OpenAIEmbeddings(), collection_name=db_name, persist_directory="{}/{}".format(self.vector_memory_path, db_name))
        else:
            self.vector_memory[db_name].add_documents(splitted_documents)

    # @staticmethod
    # def create_vector_memory_from_dir(dir_path):
    #     from langchain.document_loaders import DirectoryLoader
    #     loader = DirectoryLoader(dir_path)
    #     documents = loader.load()
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "，", "。", "\n"])
    #     splitted_documents = text_splitter.split_documents(documents)
    #     return FAISS.from_documents(splitted_documents, OpenAIEmbeddings())

    @staticmethod
    def create_vector_memory_from_texts(texts, metadatas):
        # return FAISS.from_texts(texts=texts, embedding=OpenAIEmbeddings(), metadatas=metadatas)
        return Chroma.from_texts(texts=texts, embedding=OpenAIEmbeddings(), metadatas=metadatas)

    def get_answer_from_vector_memory(self, query, doc_id, doc_name):
        # PREFIX = """You are an agent designed to answer questions about sets of documents.
        #     You have access to tools for interacting with the documents, and the inputs to the tools are questions.
        #     Sometimes, you will be asked to provide sources for your questions, in which case you should use the appropriate tool to do so.
        #     If the question does not seem relevant to any of the tools provided, just return "I don't know" as the answer.
        #     Note that you need to generate `Action` and `Action Input` or `Final Answer` etc along with your thought,
        #     Now user's input is: :"""
        db_name = "{}-{}".format(doc_id, doc_name) if doc_id else COMMON_DB_NAME
        if db_name not in self.vector_memory:
            db_name = COMMON_DB_NAME
        vectorstore_info = VectorStoreInfo(
            name="story_content",
            description="anything that you want to know about the story",
            vectorstore=self.vector_memory[db_name]
        )
        llm = ChatOpenAI(model_name="gpt-4" if config.GPT4_ENABLED else "gpt-3.5-turbo-16k", temperature=0)
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
        agent_executor = create_custom_agent(
            llm= llm,
            tools=toolkit.get_tools()[0:1],
            verbose=True,
            # prefix=PREFIX,
        )
        # agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)
        answer = agent_executor.run(query)
        return answer

KNOWLEDGE_AGENT = KnowledgeAgent()

if __name__ == "__main__":
    # knowledge_agent = KnowledgeAgent()
    # KNOWLEDGE_AGENT.load_file_to_vector_memory("the_rats_in_the_walls", "the_rats_in_the_walls", config.DOC_PATH + "/the_rats_in_the_walls.txt", "the_rats_in_the_walls.txt")
    # KNOWLEDGE_AGENT.save_vector_memory("the_rats_in_the_walls", "the_rats_in_the_walls")
    # KNOWLEDGE_AGENT.get_answer_from_vector_memory("文章的结局是什么？", "the_rats_in_the_walls", "the_rats_in_the_walls")
    # KNOWLEDGE_AGENT.get_answer_from_vector_memory("醒来后，我打了个电话给诺里斯上尉。后者听说了事情的经过后立刻赶了过来，之后发生了什么？", "the_rats_in_the_walls", "the_rats_in_the_walls")
    # KNOWLEDGE_AGENT.get_answer_from_vector_memory("等到一切准备就绪，上午11点的时候，我们所有七个人拿着明亮的探照灯与挖掘设备走进了地下室的底层，然后闩上了地窖的大门。尼葛尔曼一直跟着我们，虽然它显得有些急躁，但几个探险者都觉得没必要把它赶到门外去，但是，行走在这样一个隐约有啮齿动物出没的环境里，这只老猫的确显得有些焦虑。我们简单地介绍了那些罗马时期的铭文与留在祭坛上的未知图案，因为三个专家已经见过它们了，而且很熟悉它们的特征。而我们主要的注意力则集中在了最重要的中央祭坛上。不出一个小时，威廉•布林顿爵士就将它向后跷了起来，然后用一些我不太清楚的平衡方法保持住了祭坛的位置。 \
    #                                               这之后发生了什么？", "the_rats_in_the_walls", "the_rats_in_the_walls")
    KNOWLEDGE_AGENT.get_answer_from_vector_memory("介绍一下故事的开头，以rpg文字游戏的口吻，在主角7月22日发现一件事情之前", "the_rats_in_the_walls", "the_rats_in_the_walls")
    # knowledge_agent.add_text_to_vector_memory(["xusysh(2023-04-23):我最喜欢的颜色是灰色"])
    # knowledge_agent.add_text_to_vector_memory(["这篇`Multimodal Chain-of-Thought Reasoning in Language Models.pdf`的论文很有意思"])
    # knowledge_agent.save_vector_memory()
    # knowledge_agent.save_vector_memory()
    # knowledge_agent.get_answer_from_vector_memory("xusysh最喜欢的颜色是什么？")
    # PREFIX = """You are an agent designed to answer questions about sets of documents and the web.
    #     You have access to tools for interacting with the documents and the web.
    #     The inputs to the tools are questions.
    #     Sometimes, you will be asked to provide sources for your questions, in which case you should use the appropriate tool to do so.
    #     You will be asked for multiple questions, and you should splite the quistion into multiple parts and answer them one by one.
    #     Now the mode is [{}], and you need to thinking in english and translate only the final answer to [{}].
    #     User's input is:
    #     """.format("chat with web information and my memory", "Chinese")
    # # knowledge_agent.get_answer_from_vector_memory_and_web("用户`xusysh`最喜欢的颜色的反色是什么？", PREFIX)
    # knowledge_agent.get_answer_from_vector_memory("之前我和你聊过的那篇论文是什么？", PREFIX, "")
    # knowledge_agent.get_answer_from_vector_memory_and_web("之前我和你聊过的那篇论文是什么？然后再搜一下这篇论文相关的其他研究", PREFIX)
    # knowledge_agent.get_answer_from_vector_memory_and_web("今天是几号？", PREFIX2)