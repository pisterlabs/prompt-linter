import streamlit as st
import summarize_text
import summarize_audio
import summarize_blog
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms import OpenAI
import openai
import os
import nltk
from PIL import Image
# Loading Image using PIL
im = Image.open('summarizer icon.png')
# Adding Image to web app
st.set_page_config(page_title="QuickSumm", page_icon = im)

st.header("QuickSumm")
st.subheader("This is a web app that summarises text, audio and files.")
information=st.empty()
information.write("This uses AssemblyAI to transcribe and summarize audio, Langchain to summarize text and Llamaindex to summarize files")
placeholder = st.empty()
placeholder.image("https://miro.medium.com/max/531/0*UhKdZfJHuXJDYMbh")
def text():
    import streamlit as st
    text = st.text_area("Enter text here", height=200)
    placeholder.empty()
    information.empty()
    if st.button("Summarize"):
        if text == "":
            st.error("Please enter text to summarize")
        else:
            with st.spinner('Summarizing...'):
                new_text = summarize_text.generate_response(text)
                st.write("Summarized text:")
                st.write(new_text)



def audio():
    audio = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    type_of_summary = st.selectbox("What type of summary would you like?",("bullets","bullets_verbose","paragraph"))
    placeholder.empty()
    information.empty()
    if st.button("Summarize"):
        if audio is None:
            st.error("You have not uploaded an audio file")
        else:
            with st.spinner('Summarizing...'):
                url = summarize_audio.upload(audio)
                data, error = summarize_audio.get_transcription_result_url(url,type_of_summary)
                if error:
                    print(error)
                    st.error("Oops there as an error")
                else:
                    st.write("Summarized text")
                    print(data['text'])
                    st.write(data['summary'])
def blog():
    url = st.text_input("Enter URL here")
    placeholder.empty()
    information.empty()

    if st.button("Summarize"):
        if url == "":
            st.error("Please enter the url to summarize")
        else:
            with st.spinner('Summarizing...'):
                text = summarize_blog.summarizeURL(url,5)
                text=summarize_text.generate_response(text)
                st.write("Summarized text:")
                st.write(text)
                
def file():
    placeholder.empty()
    information.empty()
    openai.api_key = st.secrets['openAI_key']
    st.subheader("Chat with your files, powered by LlamaIndex 💬🦙")
            
    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your uploaded file!"}
        ]
    uploaded_files = st.file_uploader("Upload data file for indexing", type=["txt", "pdf", "docx", "csv", "json"])

    if uploaded_files:
        uploaded_file = uploaded_files
        # Ensure the data folder exists
        data_folder = "data"

        # Save the uploaded file to the data folder
        existing_files = os.listdir(data_folder)
        for existing_file in existing_files:
            file_to_delete = os.path.join(data_folder, existing_file)
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)

        uploaded_file = uploaded_files

        # Save the uploaded file to the data folder
        file_path = os.path.join(data_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Check the file type and handle indexing accordingly
        # if uploaded_file.type == "text/plain":
        #     # Text file, treat it as a document and add it to the index
        #     with open(file_path, "r", encoding="utf-8") as text_file:
        #         content = text_file.read()
        #         doc = Document(content=content, metadata={"filename": uploaded_file.name})
        # else:
        #     print("Nvm")

        def load_data():
            with st.spinner(text="Loading and indexing the your file – hang tight! This should take 1-2 minutes."):
                reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
                docs = reader.load_data()
                service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the document you are trained on and your job is to answer technical questions. Assume that all questions are related to the document. Keep your answers technical and based on facts – do not hallucinate features."))
                index = VectorStoreIndex.from_documents(docs, service_context=service_context)
                return index

        index = load_data()
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

        if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # If last message is not from the assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history

add_selectbox = st.sidebar.selectbox(
    'How would you like enter text?',
    ('', 'Text', 'Audio','File','Blog')
)

if add_selectbox == 'Text':
    text()

elif add_selectbox == 'Audio':
    audio()

elif add_selectbox == 'File':
    file()

elif add_selectbox == "Blog":
    blog()
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        
        footer:after {
 content: " by Sid";
}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
