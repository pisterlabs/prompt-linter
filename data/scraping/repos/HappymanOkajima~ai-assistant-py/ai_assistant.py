from io import StringIO
from langchain import LLMChain, PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tiktoken
import os
from uuid import uuid4
from pathlib import Path
from langchain.callbacks.base import BaseCallbackHandler
import mimetypes

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_model = os.environ.get("OPENAI_MODEL","gpt-3.5-turbo-16k")

def save_uploaded_file(upload, path="."):
    """
    アップロードされたファイルを保存し、そのパスを返す

    Parameters:
    upload : UploadedFile
        Streamlitでアップロードされたファイル
    path : str
        保存先のディレクトリパス
    
    Returns:
    str
        保存したファイルのパス
    """
    # ファイルを一時的に一意な名前で保存
    temp_path = Path(path) / f"temp_{uuid4().hex}"
    with temp_path.open("wb") as f:
        f.write(upload.getvalue())

    # 正しいファイル名で保存
    file_path = Path(path) / upload.name
    os.rename(temp_path, file_path)

    return str(file_path)

@st.cache_resource
def process_uploaded_file(upload):
    """
    アップロードされたファイルを処理し、その内容を返す

    Parameters:
    upload : UploadedFile
        Streamlitでアップロードされたファイル
    
    Returns:
    str
        ファイルの内容
    """
    # ファイルのMIMEタイプを推測
    mime_type, _ = mimetypes.guess_type(upload.name)
    print(mime_type)

    # ファイルの種類に応じて処理
    if mime_type is not None and (mime_type.startswith("text/") or mime_type == "application/vnd.ms-excel"):
        content = StringIO(upload.getvalue().decode('utf-8')).read()
    elif mime_type == "application/pdf":
        # ファイルを一時的に保存
        path = save_uploaded_file(upload)
        loader = PyPDFLoader(path)
        pages = loader.load()
        content = "\n".join([page.page_content for page in pages])
        print(content)

        # 一時ファイルを削除
        os.remove(path)
    else:
        content = "テキストまたはPDFファイルのみ対応しています"

    return content

@st.cache_resource
def create_summary(content):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=50)
    docs = text_splitter.create_documents(texts=[content],metadatas=[{"source": ""}])
    print(f"{len(docs)} docs")
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", streaming=True, 
                     callbacks=[StreamingStdOutCallbackHandler()], verbose=True, temperature=0)
    
    map_prompt_template = """Write 3000 words summary of the following. :

    {text}

    SUMMARY IN JAPANESE:"""
    MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    com_prompt_template = """Write 4000 words summary of the following. :

    {text}

    SUMMARY IN JAPANESE:"""
    COM_PROMPT = PromptTemplate(template=com_prompt_template, input_variables=["text"])
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                 return_intermediate_steps=False,
                                 map_prompt=MAP_PROMPT, 
                                 combine_prompt=COM_PROMPT)
    summary = chain.run(docs)
    # print(f"\nsummray = {summary}")
    return summary


@st.cache_resource
def create_prompt():

    prompt_template = """あなたはアシスタントAIでマネージャであるHumanをサポートします。
質問には、以下のテキストに含まれる情報を踏まえて回答してください。

Context:
```
{topic}    
```
  
{chat_history}

Human:{question}
AI:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question","chat_history","topic"]
    )
    
    return PROMPT
  
def extract_content_pairs(messages):
    return [f"Human: \"{messages[i].content}\",AI: \"{messages[i+1].content}\"" for i in range(0, len(messages), 2)]

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


with st.sidebar:
  st.header('AIアシスタントに相談してみよう')
  file = st.file_uploader('話題としたい資料（テキストまたはPDF）をアップロードしてください')
  if file is not None:
      topic = process_uploaded_file(file)
      tokenizer = tiktoken.encoding_for_model("gpt-4")
      tokenlen = len(tokenizer.encode(topic))
      print(f"token size = {tokenlen}")
      if tokenlen > 7000:
         topic = create_summary(topic)
    
      st.code(topic,"txt")
      st.session_state["topic"] = ChatMessage(role="assistant", content=topic)

if "topic" not in st.session_state:
    st.session_state["topic"] = ChatMessage(role="assistant", content="")
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if chat_prompt := st.chat_input("質問や相談をどうぞ"):
    st.chat_message("user").write(chat_prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        prompt = create_prompt()
        llm = ChatOpenAI(model=openai_model, openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        history = extract_content_pairs(st.session_state.messages)
        response = chain.run(question = chat_prompt, chat_history = history, topic = st.session_state["topic"].content)
        st.session_state.messages.append(ChatMessage(role="user", content=chat_prompt))
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))