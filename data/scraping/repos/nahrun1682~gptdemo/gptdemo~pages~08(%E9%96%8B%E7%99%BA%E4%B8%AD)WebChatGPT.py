import os
from dotenv import load_dotenv
import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

from libs.web_research_retriever import web_research_retriever

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
openai_api_key = os.environ["OPENAI_API_KEY"]

#langchain.verbose = False

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

#タイトルを表示
st.title('🌐(開発中)WebChatGPT')
# model_name = st.sidebar.radio(
#     "モデルを選択(1106が現在最新版):",
#     ("gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-1106","gpt-4-1106-preview"),
#     index=2)
# temperature = st.sidebar.slider("Temperature(大きいほど正確、低いほどランダム):", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

#サイドバーの折りたたみ可能なセクションを作成
with st.sidebar:
    st.header('設定')
    
    with st.expander("モデル選択"):
        model_name = st.radio(
            "モデルを選択(1106が現在最新版):",
            ("gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
            index=3
        )

    with st.expander("オプション設定"):
        temperature = st.slider(
            "Temperature(大きいほど正確、低いほどランダム):", 
            min_value=0.0, max_value=1.0, value=1.0, step=0.1
        )

    # その他のサイドバー設定
    web_mode_selection = st.radio(
    "Web検索モード(あんまダメ):",
    ('OFF', 'ON(右上のRUUNNIG終了後にOFFに戻してください)'),
    index=0
)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

if "messages_webgpt" not in st.session_state:
    st.session_state["messages_webgpt"] = [ChatMessage(role="assistant", content="なんでも聞いてね")]

for msg in st.session_state.messages_webgpt:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages_webgpt.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        # web_mode_selectionがOFFの時はChatOpenAIを使う
        if web_mode_selection == 'OFF':
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name,temperature=temperature,streaming=True, callbacks=[stream_handler])
            # print(model_name)
            response = llm(st.session_state.messages_webgpt)
            st.session_state.messages_webgpt.append(ChatMessage(role="assistant", content=response.content))
        else:
            result = web_research_retriever(prompt,model_name,temperature)
            st.session_state.messages_webgpt.append(ChatMessage(role="assistant", content=result['answer']+'\n'+'参照先\n'+result['sources']))
        