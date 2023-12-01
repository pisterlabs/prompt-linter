# .env ファイルをロードして環境変数へ反映
from dotenv import load_dotenv
load_dotenv()

# 環境変数を参照
import os
openai_key = os.getenv('openai_key')

from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader

# PDFファイルからテキストデータをロードする関数
def load_data(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# テキストデータを小さいチャンクに分割する関数
def split_text(text, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text)
    documents = [Document(page_content=t, metadata={"source": "pdf", "chunk_num": i}) for i, t in enumerate(text_chunks)]
    return documents

# Large Language Model (LLM) を初期化する関数
def initialize_llm(openai_api_key, model, temperature):
    return ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=temperature)

# 質問を生成する関数
def generate_questions(llm, chain_type, documents):
    question_chain = load_summarize_chain(llm=llm, chain_type=chain_type, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    return question_chain.run(documents)

# Retrieval-QA チェーンを作成する関数
def create_retrieval_qa_chain(openai_api_key, documents, llm):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_database.as_retriever())


# PromptTemplate をインポート
from langchain.prompts import PromptTemplate

# 以下、実際のアプリケーションの実装部分
# Streamlit アプリケーションのインポートと設定
import streamlit as st

st.set_page_config(page_title="Smart Study", layout="wide")

# アプリケーションのメイン部分
if __name__ == "__main__":
    st.title("Question Generator and Answerer")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    openai_api_key = st.text_input("Enter your OpenAI API Key")

    if uploaded_file and openai_api_key:
        # PDFからテキストデータをロード
        text_from_pdf = load_data(uploaded_file)

        # 質問生成と回答生成のためのテキストデータの分割
        documents_for_question_gen = split_text(text_from_pdf, chunk_size=1000, chunk_overlap=100)
        documents_for_question_answering = split_text(text_from_pdf, chunk_size=500, chunk_overlap=100)

        # Large Language Models (LLM) を初期化
        llm_question_gen = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.8)
        llm_question_answering = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

        # `questions`がst.session_stateに存在しない場合、初期化する
        if 'questions' not in st.session_state:
            st.session_state['questions'] = 'empty'

        # まだ質問が生成されていない場合、質問を生成
        if st.session_state['questions'] == 'empty':
            with st.spinner("Generating questions..."):
                st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type="refine", documents=documents_for_question_gen)

        # 質問を表示し、ユーザーに質問の選択を促す
        if st.session_state['questions'] != 'empty':
            st.info(st.session_state['questions'])
            st.session_state['questions_list'] = st.session_state['questions'].split('\n')

            with st.form(key='my_form'):
                st.session_state['questions_to_answers'] = st.multiselect(label="Select questions to answer", options=st.session_state['questions_list'])
                submitted = st.form_submit_button('Generate answers')
                if submitted:
                    st.session_state['submitted'] = True

            # `submitted`がst.session_stateに存在しない場合、初期化する
            if 'submitted' not in st.session_state:
                st.session_state['submitted'] = False

            # 選択された質問に対する回答を生成し、表示
            if st.session_state['submitted']:
                with st.spinner("Generating answers..."):
                    generate_answer_chain = create_retrieval_qa_chain(openai_api_key=openai_api_key, documents=documents_for_question_answering, llm=llm_question_answering)
                    for question in st.session_state['questions_to_answers']:
                        answer = generate_answer_chain.run(question)
                        st.write(f"Question: {question}")
                        st.info(f"Answer: {answer}")

    elif uploaded_file and not openai_api_key:
        st.error("Please enter your OpenAI API Key")
