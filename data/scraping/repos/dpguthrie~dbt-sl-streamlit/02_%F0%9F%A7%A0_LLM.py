# stdlib
import os

# third party
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema.output_parser import OutputParserException
from pydantic.v1.error_wrappers import ValidationError


st.set_page_config(
    page_title="dbt Semantic Layer - View Metrics",
    page_icon="🌌",
    layout="wide",
)


if "conn" not in st.session_state or st.session_state.conn is None:
    st.warning("Go to home page and enter your JDBC URL")
    st.stop()

if "metric_dict" not in st.session_state:
    st.warning(
        "No metrics found.  Ensure your project has metrics defined and a production "
        "job has been run successfully."
    )
    st.stop()


# first party
from client import get_query_results
from helpers import create_graphql_code, create_tabs, to_arrow_table
from llm.examples import EXAMPLES
from llm.prompt import EXAMPLE_PROMPT
from schema import Query


def set_openai_api_key():
    st.session_state._openai_api_key = st.session_state.openai_api_key


def set_question():
    previous_question = st.session_state.get("_question", None)
    st.session_state._question = st.session_state.question
    st.session_state.refresh = not previous_question == st.session_state._question


st.write("# LLM Query Builder")

st.markdown(
    "Input your OpenAI API Key below and ask questions abour your data.\n\n**This is highly experimental** "
    "and not meant to handle every edge case.  Please feel free to report any issues (or open up a PR to fix)."
)

api_key = st.text_input(
    label="OpenAI API Key",
    type="password",
    value=st.session_state.get("_openai_api_key", ""),
    placeholder="Enter your OpenAI API Key",
    key="openai_api_key",
    on_change=set_openai_api_key,
)

question = st.text_input(
    label="Ask a question",
    placeholder="e.g. What is total revenue?",
    key="question",
    on_change=set_question,
)

metrics = ", ".join(list(st.session_state.metric_dict.keys()))
dimensions = ", ".join(list(st.session_state.dimension_dict.keys()))

parser = PydanticOutputParser(pydantic_object=Query)

prompt_example = PromptTemplate(
    template=EXAMPLE_PROMPT,
    input_variables=["metrics", "dimensions", "question", "result"],
)

prompt = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=prompt_example,
    prefix="""Given a question involving a user's data, transform it into a structured object.
    {format_instructions}
    """,
    suffix="Metrics: {metrics}\nDimensions: {dimensions}\nQuestion: {question}\nResult:\n",
    input_variables=["metrics", "dimensions", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

if question and st.session_state.get("refresh", False):
    if st.session_state.get("_openai_api_key", None) is None:
        st.warning("Please enter your OpenAI API Key")
        st.stop()
    try:
        llm = OpenAI(
            openai_api_key=st.session_state._openai_api_key,
            model_name="gpt-3.5-turbo-instruct",
            temperature=0,
        )
    except ValidationError as e:
        st.write(e)
        st.stop()

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(metrics=metrics, dimensions=dimensions, question=question)
    try:
        query = parser.parse(output)
    except OutputParserException as e:
        st.error(e)
        st.stop()
    python_code = create_graphql_code(query)
    with st.expander("View API Request", expanded=False):
        st.code(python_code, language="python")
    payload = {"query": query.gql, "variables": query.variables}
    data = get_query_results(payload, source="streamlit-llm")
    df = to_arrow_table(data["arrowResult"])
    df.columns = [col.lower() for col in df.columns]
    st.session_state.query_llm = query
    st.session_state.df_llm = df
    st.session_state.compiled_sql_llm = data["sql"]

st.session_state.refresh = False
create_tabs(st.session_state, "llm")
