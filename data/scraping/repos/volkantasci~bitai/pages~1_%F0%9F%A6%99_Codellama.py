import streamlit as st
import requests
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from respond_beauty import make_it_beautiful
from api_config import API_HOST, API_PORT

#  List of models we can use
MODELS = [
    "Meta AI - Codellama 34b Instruct",
    "Meta AI - Codellama 34b Python",
    "Meta AI - Codellama 13b Instruct",
    "Meta AI - Codellama 13b Python",
    "Meta AI - Codellama 7b Instruct",
    "Meta AI - Codellama 7b Python",
    "OpenAI GPT-3.5-Turbo Instruct",
    "OpenAI davinci-002",
]

#  List of API URLs for each model
API_URLS = {
    "Meta AI - Codellama 34b Instruct": API_HOST + f":{API_PORT}" + "/api/v1/prediction/1f9157b9-3cbe-4bfe-852d-d0747ac0e4ff",
    "Meta AI - Codellama 34b Python": API_HOST + f":{API_PORT}" + "/api/v1/prediction/75f3c668-1261-4216-aaa8-2176ccb2ff6a",
    "Meta AI - Codellama 13b Instruct": API_HOST + f":{API_PORT}" + "/api/v1/prediction/8bd8c170-9baf-4769-9fb2-2a748749b2b2",
    "Meta AI - Codellama 13b Python": API_HOST + f":{API_PORT}" + "/api/v1/prediction/d5cb84d9-9bf0-449a-a1ba-c916877f472c",
    "Meta AI - Codellama 7b Instruct": API_HOST + f":{API_PORT}" + "/api/v1/prediction/a5a3b276-5e13-4789-9fce-b9cc64d10401",
    "Meta AI - Codellama 7b Python": API_HOST + f":{API_PORT}" + "/api/v1/prediction/5f50d315-1d0f-425c-8776-62bf0a06fad6",
    "OpenAI GPT-3.5-Turbo Instruct": API_HOST + f":{API_PORT}" + "/api/v1/prediction/b8cf4ce2-2227-4673-87d9-acbee4fcd8da",
    "OpenAI davinci-002": API_HOST + f":{API_PORT}" + "/api/v1/prediction/15ec5490-09bf-4635-8a2e-6b40a3322daa",
}

#  Create a memory object and add it to the session state
if "codellama_interface_memory" not in st.session_state:
    st.session_state.codellama_interface_memory = ConversationBufferMemory(memory_key="codellama_history")

if "codellama_model" not in st.session_state:
    st.session_state.codellama_model = "Meta AI - Codellama 34b"

if "codellama_interface_html" not in st.session_state:
    st.session_state.codellama_interface_html = False


def handle_user_input(prompt):
    """
    Handle user input and send it to the API
    """

    #  Add user input to memory
    st.session_state.codellama_interface_memory.chat_memory.add_message(HumanMessage(content=prompt))

    def query(payload):
        selected_api_url = API_URLS[st.session_state.codellama_model]
        response = requests.post(selected_api_url, json=payload)
        return response.json()

    with st.spinner("Chatting with AI..."):
        output = query({
            "question": prompt,
        })

        st.session_state.codellama_interface_memory.chat_memory.add_message(AIMessage(content=output))


def main():
    # set page config
    st.set_page_config(
        page_title="Codellama Models 🦙",
        page_icon="🦙",
        initial_sidebar_state="expanded",
    )

    #  Add title and subtitle
    st.title(":orange[bit AI] 🤖")
    st.caption(
        "bitAI powered by these AI tools:"
        "OpenAI GPT-3.5-Turbo 🤖, HuggingFace 🤗, CodeLLaMa 🦙, Replicate and Streamlit of course."
    )
    st.subheader("Code with Code🦙")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            #  List models we can use
            st.session_state.codellama_model = st.selectbox("Select a model to use Codellama:", MODELS, )

        with col2:
            st.write('<div style="height: 27px"></div>', unsafe_allow_html=True)
            second_col1, second_col2 = st.columns([2, 1])
            with second_col1:
                clear_button = st.button("🗑️ Clear history", use_container_width=True)
                if clear_button:
                    st.session_state.codellama_interface_memory.clear()

            with second_col2:
                st.session_state.codellama_interface_html = st.toggle("HTML", value=False)

    #  Set initial variables
    if "codellama_model" not in st.session_state:
        st.session_state.codellama_model = "Meta AI - Codellama 34b Instruct"

    prompt = st.chat_input("✏️ Enter your message here: ")
    if prompt:
        st.session_state.user_input = prompt
        handle_user_input(prompt)

    #  Display chat history
    for message in st.session_state.codellama_interface_memory.buffer_as_messages:
        if isinstance(message, HumanMessage):
            if st.session_state.codellama_interface_html:
                with open("templates/user_message_template.html") as user_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = user_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)
            else:
                st.chat_message("Human", avatar="🤗").write(message.content)
        elif isinstance(message, AIMessage):
            if st.session_state.codellama_interface_html:
                with open("templates/ai_message_template.html") as ai_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = ai_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)
            else:
                st.chat_message("AI", avatar="🦙").write(message.content)

    st.sidebar.caption('<p style="text-align: center;">Made by volkantasci</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
