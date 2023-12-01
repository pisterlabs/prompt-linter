import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from streamlit_chat import message

load_dotenv()


@st.cache_resource
def load_translate_prompt():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt


@st.cache_resource
def load_answer_provided_language_prompt():
    template = "You are a helpful assistant. Please provide the final output in the same language as the input text."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt


# langchain part
@st.cache_resource
def load_chain() -> LLMChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    chat = ChatOpenAI()
    # chain = LLMChain(llm=chat, prompt=load_translate_prompt(), verbose=True)
    chain = LLMChain(llm=chat, prompt=load_answer_provided_language_prompt(), verbose=True)
    return chain


# streamlit part
st.header("Simple Chat")

# show explain this is simple chat using LLMChain
st.write("LLMChain and Prompt Template Sample")

chain: LLMChain = load_chain()
language_options = ["Japanese", "English", "Chinese"]
default = "Japanese"
selected_input_language = st.selectbox("入力言語を選択してください", language_options, index=language_options.index(default))
selected_output_language = st.selectbox("出力言語を選択してください", language_options, index=language_options.index(default))


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


with st.form(key="form", clear_on_submit=True):
    user_input: str = st.text_area("You: ", "", key="input_text", placeholder="please type here")
    submit: bool = st.form_submit_button("Submit")


if submit:
    output: str = chain.run(input_language=selected_input_language, output_language=selected_output_language, text=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    wait_for_all_tracers()

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
