import os
import utils
import streamlit as st
from pathlib import Path
from streaming import StreamHandler
from functions import *
import shutil

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(page_title="ChatDocs", page_icon="📄")

class CustomDataChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    def make_retriever(self):
        embedding = OpenAIEmbeddings()
        
        docs_folder = "docs"

        # List all subdirectories in the "docs" folder
        subdirectories = [subdir for subdir in os.listdir(docs_folder) if os.path.isdir(os.path.join(docs_folder, subdir))]

        # Initialize a base vector store to merge into
        merged_db = None

        for subdir in subdirectories:
            db_path = os.path.join(docs_folder, subdir)
            
            # Load the vector store for the current directory
            current_db = FAISS.load_local(db_path, embedding)
            
            if merged_db is None:
                # If this is the first vector store, initialize the merged_db
                merged_db = current_db
            else:
                # Otherwise, merge the current vector store into the merged_db
                merged_db.merge_from(current_db)
        return merged_db

    def save_file(self, file):
        # Get the file extension from the uploaded file's name
        file_extension = file.name.split('.')[-1].lower()
        file_name = file.name[:-(len(file_extension) + 1)]
        
        if file_extension == 'pdf':
            process_pdf_file(file, file_name)
        elif file_extension == 'docx':
            process_docx_file(file, file_name)
        elif file_extension in ['xls', 'xlsx']:
            process_spreadsheet_file(file, file_name)
        elif file_extension == 'csv':
            process_csv_file(file, file_name)
        elif file_extension == 'txt':
            process_txt_file(file, file_name)
        else:
            print('File Format not supported')

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        vectordb = self.make_retriever()

        llm = ChatOpenAI(temperature=0)

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(), memory=memory, verbose=False)
        return qa_chain
    
    def delete_documents(self):
        docs_folder = "docs"

        # List all subdirectories in the "docs" folder
        subdirectories = [subdir for subdir in os.listdir(docs_folder) if os.path.isdir(os.path.join(docs_folder, subdir))]

        for subdir in subdirectories:
            delete_button = st.sidebar.button(f"Delete {subdir}", key=subdir)
            if delete_button:
                dir_path = os.path.join(docs_folder, subdir)
                try:
                    shutil.rmtree(dir_path)  # Recursively remove directory and its contents
                    st.sidebar.success(f"Deleted Document: {subdir}")
                    st.experimental_rerun()  # Rerun the Streamlit app to update the buttons
                except Exception as e:
                    st.sidebar.error(f"Failed to delete document: {subdir}. Error: {e}")

    @utils.enable_chat_history
    def main(self):
        tmp_files = os.listdir("docs")

        st.sidebar.write('Make sure uploaded file name does not contain a "." example: file.name.pdf ')
        with st.sidebar.form("my-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(label='Upload files', type=['pdf', 'docx', 'xls', 'xlsx', 'csv'], accept_multiple_files=True)
            submitted = st.form_submit_button("UPLOAD!")

            if submitted and uploaded_files is not None:
                try:
                    with st.spinner("Saving files..."):
                        for file in uploaded_files:
                            self.save_file(file)
                finally:
                    tmp_files = os.listdir("docs")

        if not tmp_files:
            st.error("Please upload documents to continue!")
            st.stop()
        else:
            self.delete_documents()
            qa_chain = self.setup_qa_chain(uploaded_files)

        
        user_query = st.chat_input(placeholder="Ask me anything!")

        
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()