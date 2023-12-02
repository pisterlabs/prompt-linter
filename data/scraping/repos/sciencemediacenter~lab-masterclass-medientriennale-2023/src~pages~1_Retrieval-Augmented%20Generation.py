import streamlit as st

from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA, create_citation_fuzzy_match_chain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import uuid

from markdown_strings import esc_format

load_dotenv()


DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_WEBSITE_URL = "https://www.sciencemediacenter.de/en/smc/smc/"
DEFAULT_QUESTION = "What is the mission of the SMC?"
DEFAULT_QA_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "Danke für die Frage!" at the end of the answer. Answer in German.
{context}
Question: {question}
Helpful Answer:"""


def create_state(name, start_value):
    if name not in st.session_state:
        st.session_state[name] = start_value


def preprocess_website(website_url):
    print("preprocess website", website_url)
    loader = WebBaseLoader(website_url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(), collection_name=uuid.uuid1().hex)
    return VectorStoreIndexWrapper(vectorstore=vectorstore)


def highlight(text, span):
    ef = lambda x: x
    return f"...{ef(text[span[0] - 20 : span[0]])}\*:red[{ef(text[span[0] : span[1]])}]\*{ef(text[span[1] : span[1] + 20])}..."
    # return text[span[0] : span[1]]


create_state("llm_temperature", DEFAULT_LLM_TEMPERATURE)
create_state("website_url", DEFAULT_WEBSITE_URL)
create_state("custom_qa_template", DEFAULT_QA_TEMPLATE)
create_state("index", None)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=st.session_state.llm_temperature)

### start page

st.title("📚 Retrieval-Augmented Generation")
st.caption("Some examples illustrating the Retrieval-Augmented Generation method")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Retrieve", "Basic Q&A", "Custom Q&A", "Context and Sources", "Settings"])

with tab1:
    st.markdown("## Retrieve")

    st.session_state.website_url = st.text_input("Website (full URL):", st.session_state.website_url)
    preprocessed = st.button("Preprocess", type="primary")

    if preprocessed:
        with st.spinner():
            st.session_state.index = preprocess_website(st.session_state.website_url)
            st.success(f"Done! #{len(st.session_state.index.vectorstore.get()['documents'])} documents in VectorDB")

with tab2:
    st.markdown("## Ask questions")
    if st.session_state.index is not None:
        question = st.text_input("Question:", DEFAULT_QUESTION, key="basic_question")
        submit = st.button("Submit", key="basic_question_submit", type="primary")

        if question and submit:
            st.info(st.session_state.index.query(question, llm=llm))

with tab3:
    st.markdown("## Customize prompt and ask questions")
    if st.session_state.index is not None:
        st.session_state.custom_qa_template = st.text_area(
            "Custom Q&A Prompt", st.session_state.custom_qa_template, height=200
        )
        question = st.text_input("Question:", DEFAULT_QUESTION, key="custom_qa_question")
        submit = st.button("Submit", key="custom_qa_question_submit", type="primary")

        if question and submit:
            custom_qa_prompt = PromptTemplate.from_template(st.session_state.custom_qa_template)

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=st.session_state.index.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": custom_qa_prompt},
            )
            result = qa_chain({"query": question})
            st.info(result["result"])

with tab4:
    st.markdown("## Context and sources")
    if st.session_state.index is not None:
        question = st.text_input("Question:", DEFAULT_QUESTION, key="context_sources_question")
        submit = st.button("Submit", key="context_sources_question_submit", type="primary")

        if question and submit:
            docs = st.session_state.index.vectorstore.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])

            fuzzy_match_chain = create_citation_fuzzy_match_chain(llm)

            results = fuzzy_match_chain.run(question=question, context=context)

            for fact in results.answer:
                st.info(esc_format(fact.fact))
                with st.expander("See sources"):
                    for span in fact.get_spans(context):
                        st.markdown(highlight(context, span))

            with st.expander("See context"):
                st.markdown(esc_format(context))

with tab5:
    st.session_state.llm_temperature = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.llm_temperature, 0.1)
