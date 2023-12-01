import streamlit as st
import os
import requests
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.chat_models import ChatOpenAI
from prompts_collection import unit_tests_generator
from langchain.schema.output_parser import StrOutputParser
import shutil
from langchain.docstore.document import Document

st.set_page_config(layout="wide", )


# Create a text input box to allow the user to enter the download folder path
download_folder = os.path.join("streamlit_apps", "data")
code_folder = 'streamlit_apps/code_files'


openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (mandatory)", type="password")
st.sidebar.divider()
openai_model = st.sidebar.selectbox("Model", options=["gpt-3.5-turbo","gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"])
openai_temperature = st.sidebar.slider(label="Temperature", min_value=0.0, max_value=1.0, step=0.05, value=1.0)
st.sidebar.divider()


openai_data = {
    "openai_api_key":openai_api_key,
    "openai_model":openai_model,
    "openai_temperature":openai_temperature
}

if 'continue_generation' not in st.session_state:
    st.session_state.continue_generation = False

st.sidebar.write("Code Splitting Parameters")
chunk_size = st.sidebar.number_input(label="Chunk Size",min_value=100, max_value=3500, step=100, value=1000)
chunk_overlap = st.sidebar.number_input(label="Chunk Overlap",min_value=0, max_value=3500, step=10, value=100)
# Create the target directory to save the code files
os.makedirs(code_folder, exist_ok=True)

# Create a dictionary variable to map file extensions to languages
file_extensions = {
    '.py': {'language': 'Python', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},
    '.js': {'language': 'JavaScript', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},
    '.jsx': {'language': 'JavaScript', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},    
    '.java': {'language': 'Java', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.JAVA, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},
    '.ts': {'language': 'TypeScript', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},    
    '.cpp': {'language': 'C++', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},    
    '.go': {'language': 'Go', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.GO, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},    
    '.rb': {'language': 'Ruby', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.RUBY, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},
    '.php': {'language': 'PHP', 'splitter': RecursiveCharacterTextSplitter.from_language(language=Language.PHP, chunk_size=chunk_size, chunk_overlap=chunk_overlap)},
    '.cs': {'language': 'C#', 'splitter': RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)}          
}

# Function to delete the temp folder and its contents
def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        # st.write(f"Deleted folder: {folder_path}")
    except Exception as e:
        st.write(f"Error deleting folder: {e}")



def convert_to_api_url(github_repo_url):

    if github_repo_url.startswith("https://github.com"):
        # Extract the username/organization and repository name
        username_repo = github_repo_url.split('github.com/')[1].split('/')

        # Extract the folder path
        folder_path = github_repo_url.split(f'github.com/{username_repo[0]}/{username_repo[1]}/')[1]

        # Construct the GitHub API URL
        api_url = f"https://api.github.com/repos/{username_repo[0]}/{username_repo[1]}/contents"
        
        return api_url


@st.cache_data
# Function to fetch and save files
def fetch_and_save_files(api_url, path=''):
    documents = []

    # Send a GET request to the GitHub API
    response = requests.get(api_url)

    # Check if the request was successful
    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item['type'] == 'file':
                file_url = item['download_url']
                file_name = item['name']

                # Check if the file extension is in the valid_ext array
                file_extension = os.path.splitext(file_name)[1]
                if file_extension in list(file_extensions.keys()):
                    # Send a GET request to the raw file URL
                    file_response = requests.get(file_url)

                    # Read the file content without saving it
                    file_content = file_response.text
                    
                    metadata = {"language":file_extensions.get(file_extension, {"language": None}).get("language"), "file_name": file_name, "file_extension": file_extension}
                    
                    document = Document(page_content=file_content, metadata=metadata)

                    documents.append(document)
                    

            elif item['type'] == 'dir':
                # Recursively fetch and save files from subfolders
                 documents.extend(fetch_and_save_files(item['url'], os.path.join(path, item['name'])))

    return documents


@st.cache_data
# Create a function to download the uploaded files
def upload_files(uploaded_files):

    for uploaded_file in uploaded_files:
        # with open(os.path.join(code_folder, uploaded_file.name), 'r', encoding='utf-8') as code_file:
        #     file_content = code_file.read()

        documents = []

        file_content = uploaded_file.read().decode("utf-8")    
        file_extension = os.path.splitext(uploaded_file.name)[1]
        file_name = uploaded_file.name
                
        metadata = {"language":file_extensions.get(file_extension, {"language": None}).get("language"), "file_name": file_name, "file_extension": file_extension}
        
        document = Document(page_content=file_content, metadata=metadata)

        documents.append(document)

    return documents


# Split all loaded docs
def split_docs(docs):
    
    splitted_docs =[]

    for doc in docs:
        splitter = file_extensions.get(doc.metadata["file_extension"], {"splitter": None}).get("splitter")

        if not splitter:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        doc_split = splitter.split_documents(documents=docs)
        splitted_docs.extend(doc_split)
    # print(len(python_docs))
    # print(splitted_docs[0])  
    return splitted_docs 

def generate_unit_tests(document, openai_data):

    model = ChatOpenAI(openai_api_key=openai_data["openai_api_key"], model=openai_data["openai_model"], temperature=openai_data["openai_temperature"])

    language = document.metadata.get("language", "")

    prompt = unit_tests_generator(language)
    chain = prompt | model | StrOutputParser()

    answer = chain.invoke({"code":document.page_content})    

    return answer

def continue_generation():
    st.session_state.continue_generation = True

def main():

    all_documents = []
    # Streamlit UI

    st.title(":test_tube: Unit Test Generator")

    st.caption(body="Upload Code File, Enter Code Block or Enter Github URL")


    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader('Upload files', accept_multiple_files=True)

    with col2:
        github_url = st.text_area("Enter Code Text or Public :globe_with_meridians: GitHub Repository URL:", placeholder="github repo url", height=50)
    if github_url.startswith("https://github.com"):
        api_url = convert_to_api_url(github_url)
        st.success(f"API URL: {api_url}")
          
    # Check if API Key and URL are provided
    if not openai_api_key:
        st.warning("Please enter the OpenAI API Key and YouTube URL before proceeding.")
        return    


    generate = st.button('Generate Unit Tests') 

    splitted_docs = []

    if generate and openai_api_key:
    
        if uploaded_files:
            all_documents = upload_files(uploaded_files)      
        elif github_url:

            if github_url.startswith("https://github.com"):
                api_url = convert_to_api_url(github_url)
                all_documents = fetch_and_save_files(api_url)
            else:                
                all_documents = [Document(page_content=github_url, metadata={"language":"", "file_name":"Code Block", "file_extension":".txt"})]
        
        splitted_docs.extend(split_docs(all_documents))

        st.warning(f"Total {len(all_documents)} document(s) have been split into {len(splitted_docs)} chunk(s) for API calls.Are you sure you want to continue?")

        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes Continue", on_click=continue_generation)
        with col2:
            st.button("Cancel")
             
    if st.session_state.continue_generation:

        if uploaded_files:
            all_documents = upload_files(uploaded_files)      
        elif github_url:
            if github_url.startswith("https://github.com"):
                api_url = convert_to_api_url(github_url)
                all_documents = fetch_and_save_files(api_url)
            else:                
                all_documents = [Document(page_content=github_url, metadata={"language":"", "file_name":"Code Block", "file_extension":".txt"})]

        splitted_docs.extend(split_docs(all_documents))

        for document in splitted_docs:
            st.write(f"FileName: {os.path.basename(document.metadata['file_name'])}")
            with st.spinner("Generating unit tests..."):
                st.code(generate_unit_tests(document, openai_data), language=document.metadata.get("language", ""))
        st.session_state.continue_generation = False
   
if __name__ == "__main__":
    main()
