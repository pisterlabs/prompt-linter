import streamlit as st
import utils.streamlit_utils as st_utils
import os
import tempfile
from langchain.document_loaders import DirectoryLoader, TextLoader
from utils.chat_with_log import handle_chat, request_summary_from_analysis
from utils.create_db import create_database
import re
import shutil

st.set_page_config(page_title="Dora the Log-Explorer", layout='wide', page_icon='🔍')
st.header('Dora the Log-File-Explorer')
desc = """Start questioning Dora about the specified log file."""
st.markdown(desc)


# Corrected: st.set_page_config should only be called once at the beginning of the script


# Define the path to your background image - this should be a URL or a local path
# If using a local path during development, it has to be relative to the `static` folder in Streamlit
background_image_path = 'url("https://i.postimg.cc/bv739fLC/image.png")'#

# Use local CSS to use a full-width background image
def set_background_image(image_path):
    """
    This function inserts custom CSS to make the provided image the background
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: {image_path};
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background image before creating any Streamlit elements
set_background_image(background_image_path)
#set_custom_font_sizes()  # Add this line

# Placeholder for the chatbot function
def chatbot_response(message):
    # Now using the cached function to get the response
    response = handle_chat(message)
    print(response)
    return f"Dora: {response}"


def save_uploaded_file(uploaded_file, directory="data", filename="uploaded_log.txt"):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)

    # Write the uploaded file to the file system
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())
    
    return file_path

def display_message(message, is_user=True):
    # User messages are grey and aligned left, chatbot responses are blue and aligned right
    bubble_color = "#262730" if is_user else "#009BEA"
    align_text = "left" if is_user else "left"
    float_text = "left" if is_user else "right"  # Ensures that the bubble floats to the correct side

    html = f"""
    <div style="margin: 5px; padding: 10px; background-color: {bubble_color}; border-radius: 15px; text-align: {align_text}; max-width: 90%; float: {float_text}; clear: both; word-wrap: break-word; overflow-wrap: break-word;">
        {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def extract_lines_with_dates(text):
    # Regex pattern to match date format "Month Abbreviation Day Time" for any month, and capture the entire line
    pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d{2}:\d{2}:\d{2}.*"
    matches = re.finditer(pattern, text)

    # Extract and return the matched lines
    return [match.group(0) for match in matches]  # group(0) contains the entire matched string

def find_line_in_file(file_path, search_text):
    """
    Search for a line in a file that contains the given text.

    Args:
    file_path (str): Path to the file.
    search_text (str): Text to search for in the file.

    Returns:
    int: The line number of the first line containing the search text, or -1 if not found.
    """
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if search_text in line:
                return line_number
    return -1

def read_file_lines(file_path, start_line, end_line):
    """Reads specified line range from a file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Adjust line numbers to be zero-indexed
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(lines))
    return lines[start_index:end_index]




st.subheader('File Upload')
if not st.session_state.get('file_processed'):
    uploaded_file = st.file_uploader("Upload a log file", type=["txt", "out"], key="log_file_uploader")

    if uploaded_file is not None:
        st.write("Uploading File")
        # Save the uploaded file and get its path
        saved_file_path = save_uploaded_file(uploaded_file)
        print(saved_file_path)

        st.write("Creating Summary")
        
        request_summary_from_analysis(saved_file_path)
        with open("summary.txt", 'r') as file:
            summary_text=file.read()
        st.session_state['summary'] = summary_text
        # Process the file content with your function
        st.write("Creating Database")
        print(os.path.basename(saved_file_path))
        #create_database(saved_file_path)
        st.write("Log-File Ready to use")
        st.session_state['file_processed'] = True
    else:
        st.write("Please upload a text file.")

tab1, tab2 = st.tabs(['Chat', 'Log File Snippets']) 
# Function to handle sending messages
def send_message():
    user_input = st.session_state.get('user_input', '')
    if user_input:  # Check if the input is not empty
        add_to_conversation(user_input, is_user=True)
        
        # Get chatbot response and add to conversation
        response = chatbot_response(user_input)
        add_to_conversation(response, is_user=False)

        with open("summary.txt", 'r') as file:
            summary_text=file.read()
        st.session_state['summary'] = summary_text

        if os.path.exists("data/uploaded_log.txt"):
            saved_file_path = "data/uploaded_log.txt"
            # Extracting and printing the relevant lines
            extracted_lines = extract_lines_with_dates(response)
            for extracted_line in extracted_lines:
                line_number = find_line_in_file(saved_file_path, extracted_line)

                # Process file and display lines
                if line_number != -1:
                    if line_number <= 10:
                        start_line, stop_line = 0, line_number
                    else:
                        start_line, stop_line = line_number - 10, line_number
                    selected_lines = read_file_lines(saved_file_path, start_line, stop_line)
                    with tab2:
                        st.text_area("Displaying lines from {} to {}:".format(start_line, stop_line), ''.join(selected_lines), height=360)
    #st.session_state['user_input'] = ""
# New function to add messages to conversation
def add_to_conversation(message, is_user=True):
    st.session_state['conversation'].append(message)



col1, col2 = st.columns([1,2])
with tab1:
    with col2:
        # Now comes the conversation part, below the summary in col2
        st.subheader("Conversation")
        if 'conversation' not in st.session_state:
            st.session_state['conversation'] = []

        # Display existing conversation
        for message in st.session_state['conversation']:
            is_user = not st.session_state['conversation'].index(message) % 2 == 0
            display_message(message, is_user)

        # Text input for user message with a callback
        st.text_input("Your message", key="user_input", on_change=send_message, value="")

        # Button to send the message
        if st.button('Send'):
            st.write("Thinking ... ")
            send_message()

    with col1:
        st.subheader("Log Summary")
        if 'summary' in st.session_state and st.session_state['summary']:
            # Split the summary text by newlines and wrap each line in <p> tags
            summary_lines = st.session_state['summary'].split('\n')
            summary_formatted = ''.join(f'<p>{line}</p>' for line in summary_lines if line)

            summary_block = f"""
                <div style="border: 0px solid #ced4da; background-color: #262730; border-radius: 0px; padding: 10px; margin-bottom: 20px; overflow-y: auto;">
                    {summary_formatted}
                </div>
            """
            st.markdown(summary_block, unsafe_allow_html=True)
        else:
            st.write("The summary will appear here after you upload and process a log file.")



