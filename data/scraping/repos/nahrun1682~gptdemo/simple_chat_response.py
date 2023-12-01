#reference:https://zenn.dev/nishijima13/articles/3b1a50b8728261
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.llms import AzureOpenAI
import openai

#streming by langcahin
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler



# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

#os.environで.envファイルにある環境変数を取得
aoai_api_type = os.environ["AOAI_API_TYPE"]
aoai_api_version = os.environ["AOAI_API_VERSION"]
aoai_api_base_url = os.environ["AOAI_API_BASE_URL"]
aoai_api_key = os.environ["AOAI_API_KEY"]
aoai_deployment_name = os.environ["AOAI_DEPLOYMENT_NAME"]
openai.api_key = os.environ["OPENAI_API_KEY"]

def simple_response_chatgpt(
    model_name: str,
    user_msg: str,
):
    """ChatGPTのレスポンスを取得

    Args:
        user_msg (str): ユーザーメッセージ。
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    print(f"test:{model_name}")
    print(f"response:{response}")
    return response

def stream_respnse_lc():
    llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
    return 
        
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 