from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.openai_functions.base import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from streamlit_chat import message

load_dotenv()


class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


# class People(BaseModel):
#     """Identifying information about all people in a text."""

#     people: Sequence[Person] = Field(..., description="The people in the text")


# langchain part


@st.cache_resource
def load_prompt() -> ChatPromptTemplate:
    prompt_msgs = [
        SystemMessage(content="You are a world class algorithm for extracting information in structured formats."),
        HumanMessage(content="Use the given format to extract information from the following input:"),
        HumanMessagePromptTemplate.from_template("{input}"),
        HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)
    return prompt


@st.cache_resource
def load_chain() -> LLMChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    chat = ChatOpenAI()
    # chain = LLMChain(llm=chat, prompt=load_translate_prompt(), verbose=True)
    chain = create_structured_output_chain(Person, llm=chat, prompt=load_prompt(), verbose=True)
    return chain


# streamlit part
st.header("OpenAI Function Chat")

# show explain this is simple chat using LLMChain
st.write("OpenAI Function Chat Sample. please provide sentence ample include name and age and favorite food")

chain: LLMChain = load_chain()


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


with st.form(key="form", clear_on_submit=True):
    user_input: str = st.text_area("You: ", "", key="input_text", placeholder="please type here")
    submit: bool = st.form_submit_button("Submit")


if submit:
    output: str = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.json(st.session_state["generated"][i].json(indent=4))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
