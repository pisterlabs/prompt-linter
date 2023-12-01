import openai
import streamlit as st
import time
from PIL import Image

# Configure site title, icon, and other
site_title = "ACIS PhD Program Chat"
site_icon = ":nerd_face:"

# Set the page title, description, and other text
page_title = "ACIS PhD Program Chat"
description = "A GPT4-powered chat assistant to answer your questions about the ACIS PhD Program at Virginia Tech's Pamplin School of Business"
instructions = 'Ask me anything about the ACIS PhD Program at Virginia Tech. I can answer questions about the program, the application process, and more.'
as_of = '' # Optional date of last update
other_text = '' # Optional sidebar text
chat_box_instructions = 'Type your questions here.'
footer_text = 'Last Updated 2023-11-20'

# Initialize the OpenAI client
client = openai

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title= site_title, page_icon= site_icon)

# Main chat interface setup
st.markdown(f"<h1 style='color: rgba(134, 31, 65, 1);'>{page_title}</h1>", unsafe_allow_html=True)
st.caption(description)
st.write(instructions)

# Display the image in the sidebar
filepath = "Vertical_VT_Full_Color_RGB.png"
image = Image.open(filepath)
st.sidebar.image(image)

if other_text != "":
    st.sidebar.write(other_text)

# Set OpenAI contants 
if st.secrets:
    if 'ASSISTANT_ID' in st.secrets:
        assistant_id = st.secrets['ASSISTANT_ID']

    if 'OPENAI_API_KEY' in st.secrets:
        openai.api_key = st.secrets['OPENAI_API_KEY']

# Initialize session state variables for file IDs and chat control
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# Define the function to process messages with citations
def process_message_with_citations(message):
    """Extract content and annotations from the message and format citations as footnotes."""
    message_content = message.content[0].text
    annotations = message_content.annotations if hasattr(message_content, 'annotations') else []
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(annotation.text, f' [{index + 1}]')

        # Gather citations based on annotation attributes
        if (file_citation := getattr(annotation, 'file_citation', None)):
            # Retrieve the cited file details (dummy response here since we can't call OpenAI)
            cited_file = {'filename': 'cited_document.pdf'}  # This should be replaced with actual file retrieval
            citations.append(f'[{index + 1}] {file_citation.quote} from {cited_file["filename"]}')
        elif (file_path := getattr(annotation, 'file_path', None)):
            # Placeholder for file download citation
            cited_file = {'filename': 'downloaded_document.pdf'}  # This should be replaced with actual file retrieval
            citations.append(f'[{index + 1}] Click [here](#) to download {cited_file["filename"]}')  # The download link should be replaced with the actual download path

    # Add footnotes to the end of the message content
    full_response = message_content.value + '\n\n' + '\n'.join(citations)
    return full_response


# Initiate Chat
st.session_state.start_chat = True

# Create a thread once and store its ID in session state
thread = client.beta.threads.create()
st.session_state.thread_id = thread.id
# st.write("thread id: ", thread.id)

# Only show the chat interface if the chat has been started
if st.session_state.start_chat:
    # Initialize the model and messages list if not already in session state
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-4-1106-preview"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    if prompt := st.chat_input(chat_box_instructions):
        # Add user message to the state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the user's message to the existing thread
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )

        # Create a run with additional instructions
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=assistant_id
        )

        # Poll for the run to complete and retrieve the assistant's messages
        while run.status != 'completed':
            time.sleep(.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )

        # Retrieve messages added by the assistant
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )

        # Process and display assistant messages
        assistant_messages_for_run = [
            message for message in messages 
            if message.run_id == run.id and message.role == "assistant"
        ]
        for message in assistant_messages_for_run:
            full_response = process_message_with_citations(message)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response, unsafe_allow_html=True)
else:
    st.write("Something is wrong. Please try again later.")


def footer(text):
    footer_html = f"""
    <style>
    .footer {{
        position: fixed;
        left: 10;
        bottom: 0;
        width: 100%;
        background-color: rgba(241, 241, 241, 0);
        color: rgba(117, 120, 123, 1);
        text-align: left;
    }}
    </style>
    <div class='footer'>
        <p>{text}</p>
    </div>
    """
    st.sidebar.markdown(footer_html, unsafe_allow_html=True)

# Example usage in your Streamlit app
footer(footer_text)
