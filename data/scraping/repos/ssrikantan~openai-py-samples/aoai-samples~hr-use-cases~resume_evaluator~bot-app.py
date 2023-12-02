import openai
import streamlit as st
import time
import sys


def init_config():
    if "deployment_name" not in st.session_state:
        sys.path.insert(0, "../../0_common_config")
        from config_data import get_deployment_name_turbo, set_environment_details_turbo

        st.session_state["deployment_name"] = get_deployment_name_turbo()
        set_environment_details_turbo()
        print(
            "deployment_name",
            st.session_state["deployment_name"],
            "\nopenai.api_base",
            openai.api_base,
            "\nopenai.api_type",
            openai.api_type,
        )
    return st.session_state["deployment_name"]


# These are the variables that need to be set depending on the demo scenario being showcased
st.title("Candidate Shortlisting")

# system_prompt = st.text_area(label="System Prompt", value="Enter the System_Prompt", height=500)


def send_message_llm(message: str, chat_history: any) -> str:
    chat_history.append({"role": "user", "content": message})
    # print(chat_history)
    response = openai.ChatCompletion.create(
        engine=init_config(), messages=chat_history, temperature=0
    )

    llm_response = response["choices"][0]["message"]["content"]
    return llm_response


def run_assistant(user_prompt, system_prompt):
    chat_history = [{"role": "system", "content": system_prompt}]
    chat_history.append({"role": "user", "content": user_prompt})
    response = send_message_llm(user_prompt, chat_history)
    st.text_area("CV Evaluation Response", response, height=500)


def send_message_llm_streaming(chat_history: any):
    full_response = ""
    message_placeholder = st.empty()
    # print(chat_history)
    for response in openai.ChatCompletion.create(
        engine=init_config(), messages=chat_history, temperature=0, stream=True
    ):
        # print('the response ----->',response)
        try:
            full_response += response.choices[0].delta.get("content", "")
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        except:
            pass
    message_placeholder.markdown(full_response)


def run_assistant_streaming(user_prompt, system_prompt):
    chat_history = [{"role": "system", "content": system_prompt}]
    chat_history.append({"role": "user", "content": user_prompt})
    response = send_message_llm_streaming(chat_history)


with st.form("my_form"):
    system_prompt = ""
    with open("JD_Selection_Criteria.txt", "r") as file:
        # system_prompt = file.read().replace('\n', '')
        system_prompt = file.read()
        st.text_area(label="System Prompt", value=system_prompt, height=500)
    user_prompt = st.text_area(
        "CV to evaluate", "paste the CV content here!!", height=500
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        # run_assistant(user_prompt,system_prompt)
        run_assistant_streaming(user_prompt, system_prompt)
