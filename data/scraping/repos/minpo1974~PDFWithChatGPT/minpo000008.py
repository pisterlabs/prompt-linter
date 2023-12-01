
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 모델 목록을 불러오는 함수 추가
def load_model_list(filename):
    with open(filename, "r") as file:
        models = file.read().splitlines()
    return models

# 모델 목록 불러오기
model_list = load_model_list("models.txt")

button(username="minpo", floating=True, width=221)

st.title("PDF Service with ChatGPT by minpojung") #타이틀

#OpenAI KEY 입력 받기
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 모델 선택 추가
selected_model = st.selectbox("모델을 선택하세요(Select a model)", model_list)

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요.(Please upload a PDF file.)",type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Stream 받아 줄 Hander 만들기
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    #Question
    st.header("업로드된 PDF에게 질문해보세요!!(Ask a question to the uploaded PDF!)")
    question = st.text_input('당신의 질문을 입력하세요.(Enter your question.)')

    if st.button('질문하기(Ask a question)'):
        with st.spinner('잠시만 기다려주세요...(Please wait...)'):
            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name=selected_model, temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
            #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            qa_chain({"query": question})
