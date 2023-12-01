import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import uuid

st.set_page_config(
    page_title="LangChain PDF App", 
    page_icon=":smiley:"
)
st.title("👋🏻😃 Upload a PDF file to get started!")
pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

def main():
    load_dotenv()
    if pdf is not None:

        st.success("PDF file uploaded!")
        
        # Initialize st.session_state if not initialized yet
        if "history" not in st.session_state:
            st.session_state.history = []
        if "prompt" not in st.session_state:
            st.session_state.prompt = ""

        prompt_placeholder = st.form('chat-form')

        #extract file
        new_pdf = PdfReader(pdf)
        page_count = len(new_pdf.pages)
        st.write("Number of pages in the PDF:", page_count)
        text = ""
        for page in new_pdf.pages:
            text += page.extract_text()

        #split text into chunks
        splitter = CharacterTextSplitter(
            # separator="\n",
            chunk_size=1000,
            chunk_overlap= 200,
            length_function=len
        )
        chunks = splitter.split_text(text)

        #embed chunks
        try:
            embedder = OpenAIEmbeddings() 
            content_base = FAISS.from_texts(chunks, embedder)
        except Exception as e:
            st.error(f"Error during embeddings: {e}")

        with prompt_placeholder:
            st.markdown("## Write your question")
            cols = st.columns((6,1))

            cols[0].text_input("Your question:", 
                                value="",
                                label_visibility="collapsed",
                                key="prompt")

            if cols[1].form_submit_button("Send", type="primary"):
                questions = st.session_state.prompt
                if questions:
                    document = content_base.similarity_search(questions) 
                    llm = OpenAI(model_name='gpt-3.5-turbo')
                    chains = load_qa_chain(llm, chain_type="stuff")
                    answer = chains.run(input_documents=document, question=questions)
                    st.session_state.history.append(f"👤 {questions}  \n💬 {answer}")
        
        # Display history at the end
        chat_placeholder = st.container()
        with chat_placeholder:
            for chat in reversed(st.session_state.history):
                st.markdown(chat)

# Call the main function if this script is run directly
if __name__ == "__main__":
    main()
