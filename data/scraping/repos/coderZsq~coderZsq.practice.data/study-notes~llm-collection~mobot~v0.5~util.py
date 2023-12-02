import config
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
import pinecone
from tavily import Client


class Util:
    def __init__(self):
        super(Util, self).__init__()
        self.configs = config.ConfigParser()

        self.ChatOpenAI = ChatOpenAI(
            openai_api_key=self.configs.get(key='openai')['api_key'],
            model_name=self.configs.get(key='openai')['chat_model'],
            temperature=0, max_tokens=2048,
        )

        self.EmbeddingOpenAI = OpenAIEmbeddings(
            openai_api_key=self.configs.get(key='openai')['api_key'],
            model=self.configs.get(key='openai')['embedding_model'],
        )

        pinecone.init(
            api_key=self.configs.get(key='pinecone')['api_key'],
            environment=self.configs.get(key='pinecone')['environment'],
        )
        self.VDBPineconeIndex = Pinecone.get_pinecone_index(self.configs.get(key='pinecone')['index'])

        self.SETavily = Client(api_key=self.configs.get(key='tavily')['api_key'])

    @staticmethod
    def concat_chat_message(system_prompt, history, message):
        # system_prompt 是对 AI 模型的角色定义和输出约束
        # history 是历史的对话信息，包括用户问题和AI回复
        # message 是当前用户提问的问题

        # 首先 system_prompt
        messages = [SystemMessage(content=system_prompt)]
        # 然后是历史对话信息
        for item in history:
            # item[0] 是用户问题， item[1] 是AI回复
            messages.append(HumanMessage(content=item[0]))
            messages.append(AIMessage(content=item[1]))
        # 最后是当前用户问题
        messages.append(HumanMessage(content=message))

        return messages

    # 文本嵌入，结果返回是向量
    def embed_query(self, text):
        return self.EmbeddingOpenAI.embed_query(
            text=text
        )

    # 输入向量，进行相似度的查询，返回最接近的文档
    def similarity_search(self, vector):
        return self.VDBPineconeIndex.query(
            top_k=3,
            include_values=False,
            include_metadata=True,
            vector=vector
        )

    # 搜索引擎对文本进行检索
    def search_engine(self, text):
        return self.SETavily.search(
            query=text,
            max_results=3
        )
