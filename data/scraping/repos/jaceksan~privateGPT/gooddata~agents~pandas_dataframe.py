import os
import json
import streamlit as st
from gooddata_sdk import GoodDataSdk
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_title_for_id
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


@st.cache_data
def ask_agent(df: pd.DataFrame, query: str) -> str:
    """
    Query Pandas agent and return the response as a string.

    Args:
        df: Data frame we want to analyze
        query: The query to ask the agent

    Returns:ch
        The response from the agent as a string.
    """
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    # Prepare the prompt with query guidelines and formatting
    with open("prompts/data_frame.txt") as fp:
        prompt = fp.read() + query

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return str(response)


def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])
    # Check if the response is a bar chart.
    elif "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            df.set_index(df.columns.values[0], inplace=True)
            # TODO - what the fuck? Looks like a Streamlit bug
            df.index.name = None
            st.bar_chart(df)
        except ValueError as e:
            st.error(f"Couldn't create DataFrame from data: {data['data']}\nError: {str(e)}")
    # Check if the response is a line chart.
    elif "line" in response_dict:
        data = response_dict["line"]
        try:
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.write(f"DF columns: {df.columns} - {df.columns.values[0]}")
            df.set_index(df.columns.values[0], inplace=True)
            # TODO - what the fuck? Looks like a Streamlit bug
            df.index.name = None
            st.line_chart(df)
        except ValueError as e:
            st.error(f"Couldn't create DataFrame from data: {data['data']}\nError: {str(e)}")
    # Check if the response is a table or if an error occurs.
    elif "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
    else:
        st.error(f"Unexpected response: {response_dict}")


@st.cache_data
def get_insights(_sdk: GoodDataSdk, workspace_id: str):
    return _sdk.insights.get_insights(workspace_id)


def render_insight_picker(_sdk: GoodDataSdk, workspace_id: str):
    insights = get_insights(_sdk, workspace_id)
    st.selectbox(
        label="Insights:",
        options=[w.id for w in insights],
        format_func=lambda x: get_title_for_id(insights, x),
        key="insight_id",
    )


@st.cache_data
def execute_insight(_sdk: GoodDataSdkWrapper, workspace_id: str, insight_id: str) -> pd.DataFrame:
    return _sdk.pandas.data_frames(workspace_id).for_insight(insight_id)


def pandas_df(sdk: GoodDataSdkWrapper, workspace_id: str):
    render_insight_picker(sdk.sdk, workspace_id)
    insight_id = st.session_state.get("insight_id")
    if insight_id:
        df = execute_insight(sdk, workspace_id, insight_id).reset_index()
        print(df)
        query = st.text_area("Enter question about this insight:")
        if st.button("Submit Query", type="primary"):
            if query:
                response = ask_agent(df, query)
                try:
                    # Poor man solution replacing single quotes
                    answer = json.loads(response.replace("'", "\""))
                except Exception as e:
                    st.warning(f"OpenAI did not return JSON: {str(e)}")
                    st.write(response)
                else:
                    write_answer(answer)

        st.dataframe(df)
