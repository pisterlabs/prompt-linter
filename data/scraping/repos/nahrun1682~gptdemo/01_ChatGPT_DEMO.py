import os
import logging
from dotenv import load_dotenv
import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

from libs.web_research_retriever import web_research_retriever

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .envファイルの読み込み
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    logger.error("環境設定ファイルが見つかりません。")
    st.error("環境設定ファイルが見つかりません。")
    st.stop()

openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API キーが設定されていません。")
    st.error("OpenAI API キーが設定されていません。")
    st.stop()

# Streamlitのセッションステートの初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="なんでも聞いてね")]

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

#タイトルを表示
st.title('🦜ChatGPT DEMO')
st.write('github：https://github.com/nahrun1682/gptdemo')
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

# チャットメッセージの表示
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            stream_handler = StreamHandler(st.empty())
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name,temperature=temperature,streaming=True, callbacks=[stream_handler])
            # print(model_name)
            response = llm(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}", exc_info=True)
            st.error(f"エラーが発生しました: {e}")
        
        
