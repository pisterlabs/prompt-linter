import json

import streamlit as st
import openai


def success_conditions(idea):
    """
    Returns success conditions for the given idea.

    :param idea: The idea to validate.
    :return: The success conditions for the given idea.
    """

    prompt = f"""
    Please list the conditions that must be met for the following ideas to be successful.

    Idea: {idea}
    """
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a logical consultant."},
            {"role": "user", "content": prompt},
        ]
    )
    return res['choices'][0]['message']['content']


def failure_risks(idea):
    """
    Returns failure risks for the given idea.

    :param idea: The idea to validate.
    :return: The failure risks for the given idea.
    """

    prompt = f"""
    Please list possible reasons why the following ideas may fail.
    Idea: {idea}
    """
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a logical consultant."},
            {"role": "user", "content": prompt},
        ]
    )
    return res['choices'][0]['message']['content']


def make_conclusion(idea, success_conditions, failure_risks):
    prompt = f"""
    Determine the probability of success of the following idea.
    Refer to success conditions and failure risks of the idea.
    Output should be JSON according to the specified schema

    Idea: {idea}
    Success conditions: {success_conditions}
    Failure risks: {failure_risks}

    Output json schema:
    - success_degree: Numerical value of 0-10. Set 10 for absolute success and 0 for absolute failure.
    - reason: Reason for the success degree.
    - other_ideas: list of other ideas to consider related to the idea
    """
    res = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a logical consultant."},
            {"role": "user", "content": prompt},
        ]
    )
    return res['choices'][0]['message']['content']


st.set_page_config(
    page_title="Idea Validator",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="collapsed")

st.title('Idea Validator')
user_api_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="Paste your OpenAI API key here",
    type="password")
openai.api_key = user_api_key

st.subheader("What's your idea?")
idea = st.text_area(
    label="Describe your idea",
    placeholder="Describe your idea here",
    max_chars=1000)

if st.button("Validate"):
    if idea and user_api_key:
        with st.spinner('Generating success conditions...'):
            conditions = success_conditions(idea)
            st.subheader("Success conditions")
            st.write(conditions)

        with st.spinner('Generating failure risks...'):
            risks = failure_risks(idea)
            st.subheader("Failure risks")
            st.write(risks)

        with st.spinner('Generating conclusion...'):
            conclusion = make_conclusion(idea, conditions, risks)
            conclusion_dict = json.loads(conclusion)
            st.subheader("Success degree")
            st.write(str(conclusion_dict["success_degree"]) + "/10")
            st.write(conclusion_dict["reason"])

            st.subheader("Other ideas")
            for other_idea in conclusion_dict["other_ideas"]:
                st.write(other_idea)
    else:
        st.error("Please enter your idea and API key.")
