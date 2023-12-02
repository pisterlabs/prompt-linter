import datetime
import streamlit as st
import openai
import pandas as pd
import numpy as np

import paperxai.credentials as credentials
import paperxai.constants as constants
from paperxai.llms import OpenAI
from paperxai.papers import Arxiv
from paperxai.report.retriever import ReportRetriever
from paperxai.prompt.base import Prompt

########## set up the page ##########
st.set_page_config(
    page_title="PaperXAI",
    page_icon="🧙‍♂️",
    layout="wide",
)
st.header("paperXai🧙")
st.subheader("A web app to explore recent papers")

if "create_report_button_clicked" not in st.session_state:
    st.session_state.create_report_button_clicked = False

if "report" not in st.session_state:
    st.session_state.report = {"topics": [], "llm_answers": [], "papers": []}

if "report_string" not in st.session_state:
    st.session_state.report_string = ""


def check_session_state_key_empty(session_state: dict, state_key: str) -> bool: # will put in utils file
    if state_key not in session_state:
        return True
    elif session_state[state_key] in ["", None]:
        return True
    return False

def click_button() -> None: # will put in utils file
    # check that model has been selected + API key entered + webpage url entered
    if (
        (check_session_state_key_empty(st.session_state, "model"))
        or (check_session_state_key_empty(st.session_state, "OPENAI_API_KEY"))
        ):
        st.session_state.create_report_button_clicked = False
    else:
        st.session_state.create_report_button_clicked = True

def define_api_key_input() -> str:
    current_key = st.session_state.get("OPENAI_API_KEY", "")
    if not (current_key in [None, ""]):
        return current_key
    elif not (credentials.OPENAI_API_KEY in [None, ""]):
        return credentials.OPENAI_API_KEY
    else:
        return ""
    
def format_topics(topics: list[str]) -> str:
    formatted_topics = ""
    for topic in topics:
        formatted_topics += "- " + topic + "\n"
    return formatted_topics

def format_html_to_markdown(html_string: str) -> str:
    html_string = (html_string.replace("<h2>", "###")
                              .replace("</h2>", "\n")
                              .replace("<h3>", "####")
                              .replace("</h3>", "\n")
                              .replace("<h4> ", "\n**")
                              .replace(" </h4>", "**\n"))
    html_string = html_string.replace("<p>", "\n").replace("</p>", "\n")
    html_string = html_string.replace("<ul>", "").replace("</ul>", "")
    html_string = html_string.replace("<li>", "-").replace("</li>", "\n")
    return html_string
    
if "OPENAI_API_KEY" in st.session_state:
    openai.api_key = st.session_state.OPENAI_API_KEY


########## sidebar ##########

with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"
        "2. Fill out the information you want to search for in the latest papers and model/pipeline parameters\n"
        "3. Chat with the model about the papers you find most interesting 💬\n"
    )
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",
        value=define_api_key_input(),
    )

    st.session_state["OPENAI_API_KEY"] = api_key_input

    model = st.sidebar.selectbox(
    "Select the model you want to use:",
    [
        "gpt-3.5-turbo",
        "gpt-4",
    ],
    index=0,
    key="model",
)
    max_papers = st.sidebar.number_input(
        "Input the # of papers you want to search through",
        step=1,
        format="%i",
        value=1000,
        key="max_papers",
    
    )

    labels_arxiv_categories = pd.read_csv(constants.ROOT_DIR+"/data/arxiv/categories.csv", sep=";")
    labels_arxiv_categories.index = labels_arxiv_categories["ID"]
    st.sidebar.multiselect(
        "Select the arXiv CS [categories](https://arxiv.org/category_taxonomy) used to search papers:",
        options=labels_arxiv_categories["ID"],
        default=["cs.AI"],
        format_func=lambda x: labels_arxiv_categories.loc[x]["Name"],
        key="arxiv_categories",
    )

    st.date_input(
        "Start date for articles to include",
        value=datetime.datetime.now() - datetime.timedelta(days=10),
        min_value=datetime.datetime.now() - datetime.timedelta(weeks=104),
        max_value=datetime.datetime.now() - datetime.timedelta(days=1),
        key="start_date",
        help="You should only include the last few days if you want the web app to run in reasonable time."
     )
    
    st.date_input(
        "End date for articles to include",
        value=datetime.datetime.now(),
        min_value=datetime.datetime.now() - datetime.timedelta(weeks=103),
        max_value=datetime.datetime.now(),
        key="end_date"
    )


    st.markdown("---")
    st.markdown("# About")
    st.markdown(
        "🧙paperXai allows you to filter through all the latest papers "
        "based off your questions. You can then chat to the model about "
        "the papers you find most interesting."
    )
    st.markdown(
        "This tool is a work in progress. "
        "You can contribute to the project on [GitHub](https://github.com/SebastianPartarrieu/paperXai/) "
        "with your feedback and suggestions💡"
    )
    st.markdown("Made by [Seb](https://twitter.com/seb_partarr)")
    st.markdown("---")

openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key. You can get one from"
        " https://platform.openai.com/account/api-keys."
    )

########## main ##########
tab1, tab2 = st.tabs(["Define", "View"])
with tab1:
    with st.form(key="qa_form"):
        query = st.text_area("What topic or precise question do you want to learn about?")
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            submit = st.form_submit_button("Add to report")
        with col2:
            clean_topics = st.form_submit_button("Remove topics")
        if submit:
            if not (query in st.session_state.report["topics"]):
                st.session_state.report["topics"].append(query)
        if clean_topics:
            st.session_state.report["topics"] = []

    st.markdown(f"**Current topics in the report:**\n"
                f"{format_topics(st.session_state.report['topics'])}")


    create_report = st.button("Create report", on_click=click_button)
    if create_report:
        if st.session_state.report["llm_answers"] == [] and st.session_state.report["topics"] != []:
            with st.spinner("Creating your report..."):
                # define language model
                openai_model = OpenAI(
                chat_model=st.session_state.model,
                embedding_model="text-embedding-ada-002",
                temperature=0.0,
                max_tokens=1000,
            )
                # get arxiv papers
                arxiv = Arxiv()
                arxiv.get_papers(categories=st.session_state.arxiv_categories,
                                max_results=int(st.session_state.max_papers))
                arxiv.write_papers()
                # load papers and compute embeddings
                df_papers = pd.read_csv(constants.ROOT_DIR + "/data/arxiv/current_papers.csv",
                                        parse_dates=["Published Date"])
                df_papers["Embeddings"] = df_papers["String_representation"].apply(
                lambda x: openai_model.get_embeddings(text=x)
            )
                papers_embeddings = df_papers["Embeddings"].values
                papers_embeddings = np.vstack(papers_embeddings)
                # save embeddings
                np.save(constants.ROOT_DIR + "/data/arxiv/papers_embeddings.npy", papers_embeddings)
                # create report
                prompter = Prompt()
                # create config
                report_config = {"title": "Streamlit arXiv digest",
                                "sections": {"section 1": {"title": "arXiv based responses",
                                                            "questions": st.session_state.report['topics']}}}
                report_retriever = ReportRetriever(
                    language_model=openai_model,
                    prompter=prompter,
                    papers_embedding=papers_embeddings,
                    df_papers=df_papers,
                    config=report_config,
                )

                report = report_retriever.create_report()
                st.session_state.report["llm_answers"] = report["arXiv based responses"]["chat_responses"]
                st.session_state.report["papers"] = report["arXiv based responses"]["papers"]
                report_string = report_retriever.format_report()
                st.session_state.report_string = report_string
                st.text("Report created, look at the view tab!")

with tab2:
    if "report_string" in st.session_state:
        if not (st.session_state.report_string in [None, ""]):
            st.markdown(
                format_html_to_markdown(report_string)
            )
    else:
        st.markdown(
            "**Please run the report creation!**"
        )