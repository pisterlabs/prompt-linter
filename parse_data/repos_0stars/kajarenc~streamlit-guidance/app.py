import guidance
import streamlit as st


@st.cache_resource
def get_prompt():
    with open("experts.handlebars", "r") as f:
        file_content = f.read()
        return file_content


if "is_authorized" not in st.session_state:
    st.session_state.is_authorized = False

if not st.session_state.is_authorized:
    passcode = st.text_input("PASSCODE", type="password")

    if passcode == st.secrets["PASSCODE"]:
        st.session_state.is_authorized = True
        st.experimental_rerun()
    elif passcode and passcode != st.secrets["PASSCODE"]:
        st.warning("Wrong passcode!")

if st.session_state.is_authorized:
    prompt_content = get_prompt()
    with st.expander("Guidance Template", expanded=False):
        st.code(prompt_content, language="handlebars")

    gpt3 = guidance.llms.OpenAI("gpt-3.5-turbo", api_key=st.secrets["OPEN_AI_API_KEY"])
    experts = guidance(prompt_content, llm=gpt3)

    user_query = st.text_input("**ANY QUESTION THAT EXPERTS COULD ANSWER:**")

    if user_query:
        with st.spinner("Thinking..."):
            executed_experts = experts(query=user_query)
            expert_names = executed_experts["expert_names"]
            st.write("**EXPERT NAMES:**")
            st.write(expert_names)
            expert_answer = executed_experts["answer"]
            st.write("**EXPERTS JOINT ANSWER:**")
            st.write(expert_answer)
