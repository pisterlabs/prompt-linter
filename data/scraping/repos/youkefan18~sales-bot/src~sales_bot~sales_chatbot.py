
import gradio as gr
from chains import SalesChain
from langchain.memory import ConversationBufferMemory


def initialize_sales_bot(vector_store_dir: str="electronic_devices_sales_qa"):
    
    global SALES_BOT
    
    SALES_BOT = SalesChain(memory=ConversationBufferMemory(memory_key="chat_history"))

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT.agent.run({"input": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    # if ans["source_documents"] or enable_chat:
    #     print(f"[result]{ans['result']}")
    #     print(f"[source_documents]{ans['source_documents']}")
    #     return ans["result"]
    # # 否则输出套路话术
    # else:
    #     return "这个问题我要问问领导"
    return ans
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="电器销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="localhost")

if __name__ == "__main__":
    # 初始化电器销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
