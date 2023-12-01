import os
from time import time
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import openai
from gooddata_sdk import GoodDataSdk
from gooddata_sdk.catalog.data_source.declarative_model.physical_model.pdm import CatalogScanResultPdm
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_name_for_id

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


def load_chain(model_name: str) -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=0, model_name=model_name)

    chain = ConversationChain(llm=llm)
    return chain


@st.cache_data
def list_data_sources(_sdk: GoodDataSdk):
    return _sdk.catalog_data_source.list_data_sources()


def render_data_source_picker(_sdk: GoodDataSdk):
    data_sources = list_data_sources(_sdk)
    if "data_source_id" not in st.session_state:
        st.session_state["data_source_id"] = data_sources[0].id
    st.selectbox(
        label="Data sources:",
        options=[x.id for x in data_sources],
        format_func=lambda x: get_name_for_id(data_sources, x),
        key="data_source_id",
    )


def render_result_type_picker():
    if "result_type" not in st.session_state:
        st.session_state["result_type"] = "SQL statements"
    st.selectbox(
        label="Result type:",
        options=["SQL statements", "dbt models"],
        key="result_type",
    )


@st.cache_data
def get_supported_models() -> list[str]:
    return [m["id"] for m in openai.Model.list()["data"]]


def render_openai_models_picker():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-0613"
    models = get_supported_models()
    st.selectbox(
        label="OpenAI model:",
        options=models,
        key="openai_model",
    )


@st.cache_data
def scan_data_source(_sdk: GoodDataSdk, _logger, data_source_id: str) -> CatalogScanResultPdm:
    _logger.info(f"scan_data_source {data_source_id=} START")
    start = time()
    result = _sdk.catalog_data_source.scan_data_source(data_source_id)
    duration = int((time() - start)*1000)
    _logger.info(f"scan_data_source {data_source_id=} duration={duration}")
    return result


def generate_list_of_source_tables(pdm: CatalogScanResultPdm) -> str:
    result = f"The source tables are:\n"
    # Only 5 tables for demo purposes, large prompt causes slowness
    # TODO - train a LLM model with PDM model once and utilize it
    i = 1
    for table in pdm.pdm.tables:
        column_list = ", ".join([c.name for c in table.columns])
        result += f"- \"{table.path[-1]}\" with columns {column_list}\n"
        i += 1
        if i >= 5:
            break

    return result


@st.cache_data
def ask_question(_logger, request: str) -> str:
    start = time()
    chain = load_chain(st.session_state.openai_model)
    result = chain.run(input=request)
    duration = int((time() - start)*1000)
    _logger.info(f"OpenAI query duration={duration}")
    return result


def any_to_star_model(sdk: GoodDataSdkWrapper, logger):
    columns = st.columns(3)
    with columns[0]:
        render_data_source_picker(sdk.sdk)
    with columns[1]:
        render_result_type_picker()
    with columns[2]:
        render_openai_models_picker()

    try:
        if st.button("Generate", type="primary"):
            data_source_id = st.session_state["data_source_id"]
            pdm = scan_data_source(sdk.sdk, logger, data_source_id)

            with open("prompts/any_to_star_schema.txt") as fp:
                prompt = fp.read()
            request = prompt + f"""

Question:
{generate_list_of_source_tables(pdm)}
Generate {st.session_state.result_type}.
"""
            with open("tmp_prompt.txt", "w") as fp:
                fp.write(request)

            logger.info(f"OpenAI query START")
            output = ask_question(logger, request)
            st.write(output)
    except openai.error.AuthenticationError as e:
        st.write("OpenAI unknown authentication error")
        st.write(e.json_body)
        st.write(e.headers)
