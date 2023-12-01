from dataclasses import dataclass
from typing import Literal
import streamlit as st
from langchain.callbacks import get_openai_callback
import streamlit.components.v1 as components
from functions import *

st.set_page_config(
    page_title="CHild"

)


if "conversation" not in st.session_state:
    st.session_state.conversation = None 
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = None
@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    css_file = "static/styles.css"
    with open(css_file, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        
        if not human_prompt:
            llm_response = "Please upload data or enter a link"
        else:
            try:
                llm_response = st.session_state.conversation({
                    'question': human_prompt
                })

                st.session_state.history.append(
                    Message("human", human_prompt)
                )
                st.session_state.history.append(
                    Message("ai", llm_response["answer"])  
                )

                # Log the response to the Streamlit console for debugging
                print("LLM Response:", llm_response)

            except Exception as e:
                # Print the detailed error message to the Streamlit console
                llm_response = "An error occurred during processing."
            
        st.session_state.human_prompt = ""  # Clear the user input after processing


            
        

load_css()
initialize_session_state()

st.title("Chatbot 🤖")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row 
            {'' if chat.origin == 'ai' else 'row-reverse'}">
            <div class="chat-bubble
            {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)





    
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )
with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your data",accept_multiple_files=True)
        # st.subheader("YouTube video or web site link")
        # links =  st.text_input("Link")
        # links = links.split(',')
        if st.button("Progress"):
            with st.spinner("Progressing"):
                # get pdf text
                raw_text = ""
                st.session_state['uploaded_text'] = get_text_from_file(uploaded_files)
                # get text from url
                url_text = ""
                # if len(links) > 0: 
                #     url_text += load_url(links)
                # merge  url text and raw text
                st.session_state['uploaded_text']+=url_text# get chunks
                chunks = get_text_chunks(st.session_state['uploaded_text'])
                # get vectorstore 
                vectorstore = get_vectorstore(chunks=chunks)
                # get conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            
            
components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
</script>
""", 
    height=0,
    width=0,
)
