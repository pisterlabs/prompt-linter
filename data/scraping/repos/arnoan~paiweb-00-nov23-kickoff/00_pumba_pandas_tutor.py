import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage, ChatMessage

from utils import make_streamlit_ui_clean
from utils import StreamHandler

make_streamlit_ui_clean()
st.title("Pumba – Der freundliche Pandas Tutor")

SYSTEM_MESSAGE_GERMAN = [
    SystemMessage(
        content="You are a helpful assistant that is an expert in the Python Pandas data manipulation library."
                "If possible, always include sample Python code in your responses."
                "And you always provide explanations in GERMAN, even if the question is asked"
                "in a different language."
                "Also, always use informal language, that is address the user with 'Du' instead of 'Sie'."
                "End your response with a question to keep the conversation going."
    ),
]

SYSTEM_MESSAGE_ENGLISH = [
    SystemMessage(
        content="You are a helpful assistant that is an expert in the Python Pandas data manipulation library."
                "If possible, always include sample Python code in your responses."
                "And you always provide explanations in ENGLISH, even if the question is asked"
                "in a different language."
                "End your response with a question to keep the conversation going."
    ),
]

# Initialize chat history, if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hallo, wie kann ich dir helfen?"),
    ]

# region SIDEBAR SETUP

st.sidebar.image(
    "avatars/friendly_warthog.png",
    caption="Pumba, the friendly tutor",
    use_column_width=True,
    # width=150,
)

# Widget to select the language
languages = ["ENGLISH", "GERMAN"]
selected_language = st.sidebar.selectbox(
    "Select Language", languages, key="selected_language"
)

clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state["messages"] = []

# endregion SIDEBAR SETUP


BOT_AVATAR = "avatars/friendly_warthog.png"
HUMAN_AVATAR = "avatars/human.png"

# Render chat history to UI
for msg in st.session_state.messages:
    image = HUMAN_AVATAR if msg.type == "human" else BOT_AVATAR
    st.chat_message(msg.type, avatar=image).write(msg.content)

# Get user input
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("human", avatar=HUMAN_AVATAR).write(prompt)

    # Get response from model
    with st.chat_message("ai", avatar=BOT_AVATAR):
        stream_handler = StreamHandler(st.empty())

        # Set up the LangChain model
        llm = ChatOpenAI(model_name="gpt-4", streaming=True, callbacks=[stream_handler])

        # Determine language to use
        SYSTEM_MESSAGE_SELECTED = (
            SYSTEM_MESSAGE_ENGLISH
            if selected_language == "ENGLISH"
            else SYSTEM_MESSAGE_GERMAN
        )

        # Obtain response from model (pass in the entire conversation history as context)
        response = llm(SYSTEM_MESSAGE_SELECTED + st.session_state.messages)

        # Add response to chat history
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )
