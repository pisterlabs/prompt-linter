from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import OnlinePDFLoader  # for loading the pdf
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)


st.set_page_config(
    page_title="LangChain: Chat with online PDF Q&A", page_icon="🦜")
st.title("🦜 LangChain: LangChain: Chat with online PDF Q&A")


@st.cache_resource
def configure_qa_chain(pdf_addr):
    st.info("Loading online PDF...")
    loader = OnlinePDFLoader(pdf_addr)

    if not loader:
        st.info("Please upload PDF documents to continue.")
        st.stop()

    st.info("Split text recursive...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    st.info("OpenAIEmbeddings...")
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(docs, embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Custom Prompts
    PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:
    Answer reply in zh-tw:"""

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    # 3. Querying
    st.info("RetrievalQA...")
    llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0613")
    retriever = docsearch.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    return qa


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

online_pdf_addr = st.sidebar.text_input("Your online PDF address")
if not online_pdf_addr:
    st.info("Please input online PDF address to continue.")
    st.stop()


qa_chain = configure_qa_chain(online_pdf_addr)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# show paper agenda first.

st.chat_message("user").write("本篇論文的摘要是什麼?")
agenda = qa_chain.run({"query": "本篇論文的摘要是什麼?"})
st.write(agenda)

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        cb = PrintRetrievalHandler(st.container())
        response = qa_chain.run({"query": user_query}, callbacks=[cb])
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
