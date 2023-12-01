import gradio as gr
import random
import time
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

global SALES_BOT


# 从GPT4获取语料数据
# Promote：
# 你是中国顶级的英语教育培训销售，你们主营的业务是出国英语培训，现在培训职场销售新人，请给出100条实用的销售话术。
#
# 每条销售话术以如下格式给出：
# [客户问题]
# [销售回答]
#
# 保存到 data.txt 中
def prepare_data():
    with open("data.txt") as f:
        real_estate_sales = f.read()

    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local("real_estates_sale")


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt4", temperature=0)

    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True
    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="出国英语培训",
        retry_btn=None,
        undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化语料数据
    # prepare_data()

    # 初始化销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
