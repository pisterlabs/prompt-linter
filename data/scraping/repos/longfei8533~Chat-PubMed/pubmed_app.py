import streamlit as st
import openai
import json
import pubmed
from Utils import Functions, chat
from prompt import Prompt

# https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/

st.set_page_config(page_title="Chat PubMed", page_icon=":robot:")
st.header("Chat PubMed")


with st.sidebar:
    st.title("Chat PubMed")
    openai.api_base = st.text_input(
        "Enter OpenAI API base URL:", value="https://api.chatanywhere.cn/v1"
    )
    openai.api_key = st.text_input("Enter OpenAI API token:", type="password")
    if openai.api_key:
        st.success("Credentials Saved.")
    else:
        st.warning("Enter your OpenAI API token above.")
    tools_use_pub = st.checkbox("Enable retrieval PubMed articles", value=True)
    if tools_use_pub:
        functions = Functions().functions_list
    else:
        functions = None


if "query_history" not in st.session_state:
    st.session_state["query_history"] = []
    st.session_state["user_input"] = []
    st.session_state["assistant_response"] = []
    st.session_state.query_history.append(
        {
            "role": "system",
            "content": Prompt().system_prompt,
        }
    )
else:
    for i in range(len(st.session_state.assistant_response)):
        with st.chat_message("user"):
            st.write(st.session_state.user_input[i])
        with st.chat_message("assistant"):
            st.write(st.session_state.assistant_response[i])


def get_text():
    input_text = st.chat_input(placeholder="Your message")
    return input_text


user_input = get_text()


if user_input:
    st.session_state.user_input.append(user_input)
    st.session_state.query_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_message = chat(
                messages=st.session_state["query_history"],
                functions=functions,
                model="gpt-4-0613",
            )
            response_message = response_message.choices[0].message
            if response_message.get("function_call"):
                function_name = response_message["function_call"]["name"]

                if function_name == "search_pubmed":
                    function_args = json.loads(
                        response_message["function_call"]["arguments"]
                    )
                    function_response = pubmed.PubMedAPIWrapper(
                        query=function_args.get("query")
                    ).run()

                st.session_state.query_history.append(response_message)
                st.session_state.query_history.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )

                response_message = chat(
                    messages=st.session_state["query_history"],
                    functions=Functions().functions_list,
                    model="gpt-4-0613",
                )
                response_message = response_message.choices[0].message

            st.session_state.query_history.append(response_message)

            st.session_state.assistant_response.append(response_message["content"])
            st.write(response_message["content"])
