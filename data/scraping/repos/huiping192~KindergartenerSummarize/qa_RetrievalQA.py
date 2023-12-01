import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.faiss import FAISS
import langchain

with st.sidebar:
    llm_type = st.selectbox(
        'LLM選んでください',
        ('gpt-3.5-turbo', 'llama2'))

load_dotenv()

langchain.verbose = True

if llm_type == "llama2":
    llm = Ollama(base_url="http://localhost:11434", model="llama2")
else:
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

@st.cache_resource
def load_data(llm_type):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    loader = TextLoader("./content.txt")
    if llm_type == "llama2":
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
    else:
        embeddings = OpenAIEmbeddings()
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    return FAISS.from_documents(docs, embeddings)


vector_store = load_data(llm_type=llm_type)
prompt_template_qa = """あなたは親切で優しいアシスタントです。丁寧に、日本語でお答えください！
もし以下の情報が探している情報に関連していない場合は、わかりませんと答えてください。

{context}

質問: {question}
回答（日本語）:"""

prompt_qa = PromptTemplate(
        template=prompt_template_qa,
        input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": prompt_qa}
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=vector_store.as_retriever(),
                                    chain_type_kwargs=chain_type_kwargs)

st.title("幼稚園QA")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

user_query = st.chat_input("質問を入力してください")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer = chain.run(user_query)
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
