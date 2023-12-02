import base64
from typing import List
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
from io import StringIO
from IPython.display import Markdown
import json
import os
import PyPDF2
import streamlit as st
from typing import List
import itertools
import random
import pypdf
from llama_index import (
    LLMPredictor,
    LangchainEmbedding,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.schema import Document
from langchain.llms import OpenAI
from langchain.chains import QAGenerationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools


HO3_RAW_DATA_PATH = "../data/HO3_sample.pdf"

ho3_directory = "../ho3_sample_policy_meta_index"
doi_directory = "../tx_doi_index"
uniform_building_codes = "../_property_index_storage"


ss = st.session_state
if "debug" not in ss:
    ss["debug"] = {}


@st.cache_data
def load_docs(file_path):
    all_text = ""
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    all_text += text
    return all_text


def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write("")
    if line:
        st.tabs([" "])
    for _ in range(next_n):
        st.write("")


def ui_info():
    st.write(
        "This sandbox is powered by :statue_of_liberty:**LibertyGPT**, 🦜[LangChain](https://langchain-langchain.vercel.app/docs/get_started/introduction.html) and :llama:[Llama-Index](https://gpt-index.readthedocs.io/en/latest/index.html)",
        unsafe_allow_html=True,
    )


def wrap_doc_in_html(docs: List[str]) -> str:
    """Wraps each page in document separated by newlines in <p> tags"""
    # Convert to langchain schema
    docs_ = [Document.to_langchain_format(doc) for doc in docs]
    text = [doc.page_content for doc in docs_]
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])


@st.cache_data
# function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def get_embed_model():
    return LangchainEmbedding(OpenAIEmbeddings())


def get_llm(temperature=0):
    return ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")


def get_llm_predictor(temperature=0):
    return LLMPredictor(ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo"))


@st.cache_resource
def initialize_index(storage_directory):
    llm = get_llm()
    embed_model = get_embed_model()

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )

    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=storage_directory),
        service_context=service_context,
    )
    return index


ho3_index = initialize_index(storage_directory=ho3_directory)
doi_index = initialize_index(storage_directory=doi_directory)
bldg_code_index = initialize_index(storage_directory=uniform_building_codes)


@st.cache_data(show_spinner=False, experimental_allow_widgets=True)
def generate_eval(text, N, chunk):
    if "eval_set" not in ss:
        ss["eval_set"] = ""
    n = len(text)
    starting_indices = [random.randint(0, n - chunk) for _ in range(N)]
    sub_sequences = [text[i : i + chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(get_llm(temperature=0.75))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            print("Creating Question:", i + 1)
        except:
            print("Error generating question")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


def ui_index():
    index_choice = ["Simple Policy Retriever", "Simple DOI Code Retriever"]
    st.selectbox("Choose a Retriever Model", index_choice, key="index")


def display_output():
    output_placeholder = st.empty()
    with output_placeholder.container():
        output = ss.get("output", "")
        st.markdown(output)
        st.divider()
    with st.container():
        st.markdown("### Source:")
        source_output = ss.get("source", "")
        components.html(wrap_text_in_html(source_output), height=250, scrolling=True)


def output_source(s: str):
    s = s.replace("$", r"\$")
    new_source = f"{s}\n"
    ss["source"] = f"{new_source}"


def output_add(q: str, a: str):
    q = q.replace("$", r"\$")
    a = a.replace("$", r"\$")
    new_output = f"#### {q}\n{a}\n"
    ss["output"] = f"{new_output}"


def b_index():
    if st.button(
        "Initialize Model", key="init_index_1", type="primary", use_container_width=True
    ):
        selected_index = ss.get("index", "")
        if selected_index == "Simple Policy Retriever":
            ss["llama_index"] = initialize_index(storage_directory=(ho3_directory))
        elif selected_index == "Simple DOI Code Retriever":
            ss["llama_index"] = initialize_index(storage_directory=(doi_directory))


def ask():
    question = ss.get("question", "")
    selected_index = ss.get("index", "")
    index = ss.get("llama_index", {})
    if selected_index == "Simple Policy Retriever":
        response = index.as_query_engine().query(question)
        q = question.strip()
        a = str(response)
        s = (
            str(response.source_nodes[0].node.get_metadata_str())
            + "\n\n"
            + "\n\n".join([str(node.node.get_text()) for node in response.source_nodes])
        )
        ss["answer"] = a
        output_add(q, a)
        output_source(s)
    elif selected_index == "Simple DOI Code Retriever":
        response = index.as_query_engine().query(question)
        q = question.strip()
        a = str(response)
        s = str(response.source_nodes[0].node.get_text())
        ss["answer"] = a
        output_add(q, a)
        output_source(s)


def b_ask():
    prompt_placeholder = st.form("chat-form")
    with prompt_placeholder:
        cols = st.columns((9, 1))
        cols[0].text_input(
            "question",
            placeholder="Send a message",
            help="",
            label_visibility="collapsed",
            key="question",
        )
        cols[1].form_submit_button(
            "Submit", type="primary", on_click=ask, use_container_width=True
        )


def b_get_eval_set():
    if st.checkbox("Get Q&A Sets"):
        loaded_text = load_docs(HO3_RAW_DATA_PATH)
        # Use the generate_eval function to generate question-answer pairs
        num_eval_questions = 5  # Number of question-answer pairs to generate
        ss["eval_set"] = generate_eval(loaded_text, num_eval_questions, 1000)
        e = ss["eval_set"]
        # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(ss.get("eval_set")):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        eval_add(e)


def eval_add(e):
    if "eval_set" not in ss:
        ss["eval_set"] = ""
    ss["eval_set"] = e


def b_clear():
    if st.button("clear output", type="secondary", use_container_width=True):
        ss["output"] = ""
        ss["source"] = ""
        generate_eval.clear()


def main():
    # Add custom CSS
    st.markdown(
        """
        <style>
        
        #MainMenu {visibility: hidden;
        # }
            footer {visibility: hidden;
            }
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: green;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
            

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">LibertyGPT Sandbox</h1>
        <sup style="margin-left:5px;font-size:small; color: green;">private beta</sup>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ui_info()

    with st.sidebar:
        ui_spacer(2)
        with st.expander("**Settings**"):
            b_index()
            b_clear()
            ui_spacer(1)
            ui_index()
        ui_spacer(2)
        b_get_eval_set()
    b_ask()
    ui_info()
    cols2 = st.columns(2, gap="small")
    with cols2[0]:
        # with st.expander("**Document Excerpts**"):
        # placeholder = st.empty()
        # with placeholder.expander("**Query Results**", expanded=False):
        st.write(display_output())
    with cols2[1]:
        with st.expander("**Original Source Documents**"):
            displayPDF(HO3_RAW_DATA_PATH)


if __name__ == "__main__":
    main()
