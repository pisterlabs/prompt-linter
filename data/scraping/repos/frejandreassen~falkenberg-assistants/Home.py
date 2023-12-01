import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["openai_api_key"])

# User input fields
assistant_id = st.secrets["assistant_id"]
if 'assistant' not in st.session_state:
    st.session_state.assistant = client.beta.assistants.retrieve(assistant_id)
assistant = st.session_state.assistant

assistant_files = client.beta.assistants.files.list(
        assistant_id=assistant_id
    )

# Streamlit app layout
st.write(assistant_id)
st.title("AI Assistant Update Form")

name = st.text_input("Namn", assistant.name)
description = st.text_area("Beskrivning", assistant.description)
instructions = st.text_area("Instruktioner (dold för användare)", assistant.instructions)

# Update logic
if st.button("Update Assistant"):
    # Initialize OpenAI client

    try:
        # Update the assistant
        updated_assistant = client.beta.assistants.update(
            assistant_id,
            name=name,
            description=description,
            instructions=instructions
        )
        st.success("Assistant updated successfully!")
        st.session_state.assistant = updated_assistant
    except Exception as e:
        st.error(f"An error occurred: {e}")



# Display each file with a delete button
st.header("Assistant files")
st.write("The assistant's custom knowledge base")
st.divider()
for assistant_file in assistant_files.data:
    file = client.files.retrieve(assistant_file.id)
    # st.write(file)
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.write(file.filename)
    file_size_mb = file.bytes / (1024 ** 2)
    col2.write(f"File size: {file_size_mb:.2f} MB")
    if col3.button('🗑️', key=file.id):
        # Logic to delete the file
        client.beta.assistants.files.delete(
            assistant_id=assistant_id,
            file_id=file.id
        )
        client.files.delete(file.id)
        st.rerun()
st.divider()

allowed_extensions = [
    "c", "cpp", "csv", "docx", "html", "java", 
    "json", "md", "pdf", "php", "pptx", "py", 
    "rb", "tex", "txt"
]

uploaded_file = st.file_uploader("Upload file", type=allowed_extensions)
if uploaded_file is not None:
    file = client.files.create(
        file=uploaded_file,
        purpose="assistants"
    )
    client.beta.assistants.files.create(
        assistant_id=assistant_id, 
        file_id=file.id
    )
